/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <errno.h>
#include <fstream>
#include <iostream>
#include <string>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <assert.h>
#include "NvUtils.h"
#include "NvCudaProc.h"

#include "videodec.h"
#include "nvosd.h"

#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define CHUNK_SIZE 4000000
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
        !buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
        (buffer_ptr[2] == 1))

#define BORDER_WIDTH 5

#define FIRST_CLASS_CNT 1

using namespace std;


/**
   * Read the input NAL unit for h264/H265/Mpeg2/Mpeg4 decoder.
   *
   * @param stream            : Input stream
   * @param buffer            : NvBuffer pointer
   * @param parse_buffer      : parse buffer pointer
   * @param parse_buffer_size : chunk size
   */
static int
read_decoder_input_nalu(ifstream * stream, NvBuffer * buffer,
        char *parse_buffer, streamsize parse_buffer_size)
{
    /* Length is the size of the buffer in bytes */
    char *buffer_ptr = (char *) buffer->planes[0].data;
    char *stream_ptr;
    bool nalu_found = false;
    streamsize bytes_read;
    streamsize stream_initial_pos = stream->tellg();

    stream->read(parse_buffer, parse_buffer_size);
    bytes_read = stream->gcount();

    if (bytes_read == 0)
    {
        return buffer->planes[0].bytesused = 0;
    }

    /* Find the first NAL unit in the buffer */
    stream_ptr = parse_buffer;
    while ((stream_ptr - parse_buffer) < (bytes_read - 3))
    {
        nalu_found = IS_NAL_UNIT_START(stream_ptr) ||
                    IS_NAL_UNIT_START1(stream_ptr);
        if (nalu_found)
        {
            break;
        }
        stream_ptr++;
    }

    /* Reached end of buffer but could not find NAL unit */
    if (!nalu_found)
    {
        cerr << "Could not read nal unit from file. EOF or file corrupted"
            << endl;
        return -1;
    }

    memcpy(buffer_ptr, stream_ptr, 4);
    buffer_ptr += 4;
    buffer->planes[0].bytesused = 4;
    stream_ptr += 4;

    /* Copy bytes till the next NAL unit is found */
    while ((stream_ptr - parse_buffer) < (bytes_read - 3))
    {
        if (IS_NAL_UNIT_START(stream_ptr) || IS_NAL_UNIT_START1(stream_ptr))
        {
            streamsize seekto = stream_initial_pos +
                    (stream_ptr - parse_buffer);
            if(stream->eof())
            {
                stream->clear();
            }
            stream->seekg(seekto, stream->beg);
            return 0;
        }
        *buffer_ptr = *stream_ptr;
        buffer_ptr++;
        stream_ptr++;
        buffer->planes[0].bytesused++;
    }

    /* Reached end of buffer but could not find NAL unit */
    cerr << "Could not read nal unit from file. EOF or file corrupted"
            << endl;
    return -1;
}

/**
 * Read the input chunks for h264/H265/Mpeg2/Mpeg4 decoder.
 *
 * @param stream : Input stream
 * @param buffer : NvBuffer pointer
 */
static int
read_decoder_input_chunk(ifstream * stream, NvBuffer * buffer)
{
    /* Length is the size of the buffer in bytes */
    streamsize bytes_to_read = MIN(CHUNK_SIZE, buffer->planes[0].length);

    stream->read((char *) buffer->planes[0].data, bytes_to_read);
    /* NOTE: It is necessary to set bytesused properly, so that decoder knows how
       many bytes in the buffer are valid */
    buffer->planes[0].bytesused = stream->gcount();
    return 0;
}

/**
 * Exit on error.
 */
static void
abort(context_t *ctx)
{
    ctx->got_error = true;
    ctx->dec->abort();
}

/**
 * Set the text osd parameters
 */
static void
set_text(context_t* ctx)
{

    ctx->textParams.display_text = ctx->osd_text ? : strdup("nvosd overlay text");
    ctx->textParams.x_offset = 30;
    ctx->textParams.y_offset = 30;
    ctx->textParams.font_params.font_name = strdup("Arial");
    ctx->textParams.font_params.font_size = 18;
    ctx->textParams.font_params.font_color.red = 1.0;
    ctx->textParams.font_params.font_color.green = 0.0;
    ctx->textParams.font_params.font_color.blue = 1.0;
    ctx->textParams.font_params.font_color.alpha = 1.0;
}

/**
 * Read the rectangle OSD information from OSD file
 * Below is a sample of the OSD info for frame 1

    frame:1 class num:0 has rect:5
        x,y,w,h:0.556251 0.413043 0.0390625 0.0489113
        x,y,w,h:0.595312 0.366848 0.0546875 0.0923913
        x,y,w,h:0.090625 0.364132 0.2218750 0.2010872
        x,y,w,h:0.323438 0.413043 0.0984375 0.1114133
        x,y,w,h:0.403125 0.418478 0.0406252 0.0760875
 */
static void
get_rect(context_t *ctx)
{
    unsigned int i;
    string line;
    int frameNum = -1;
    int classNum = 0;
    int rect_id = 0;
    ctx->g_rect_num = 0;

    if (ctx->osd_file->is_open())
    {
        while (getline(*ctx->osd_file, line))
        {
            std::string frameNumPrefix = "frame:";
            std::string rectNumPrefix = " has rect:";
            std::string classNumPrefix = " class num:";
            std::string xywhPrefix = "x,y,w,h:";
            if (line.compare(0, frameNumPrefix.size(), frameNumPrefix) == 0)
            {
                string strFrameNum = line.substr(6, line.find(rectNumPrefix) - 6);
                if (frameNum >= 0)
                {
                    assert(frameNum == atoi(strFrameNum.c_str()));
                }
                frameNum = atoi(strFrameNum.c_str());
                string strClassNum =
                            line.substr(line.find(classNumPrefix) + classNumPrefix.size(),
                                        line.find(classNumPrefix) + classNumPrefix.size() + 1);
                classNum = atoi(strClassNum.c_str());
                string strRectNum =
                            line.substr(line.find(rectNumPrefix) + rectNumPrefix.size(),
                                        line.size());
                ctx->g_rect_num = atoi(strRectNum.c_str());

                if (log_level >= LOG_LEVEL_DEBUG)
                {
                    cout << "frameNum: " << frameNum
                         << " class num: " << classNum
                         << " Rect Num: " << ctx->g_rect_num << endl;
                }
            }
            else if (std::string::npos != line.find(xywhPrefix))
            {
                string xywh = line.substr(line.find(xywhPrefix) + xywhPrefix.size(),
                                          line.size());
                for (i = 0; i < xywh.size(); ++i)
                {
                    if (isspace(xywh.at(i))) xywh.replace(i, 1, 1, ':');
                }
                string x = xywh.substr(0, xywh.find(":"));
                xywh = xywh.substr(x.size() + 1);
                string y = xywh.substr(0, xywh.find(":"));
                xywh = xywh.substr(y.size() + 1);
                string w = xywh.substr(0, xywh.find(":"));
                xywh = xywh.substr(w.size() + 1);
                string h = xywh.substr(0);
                ctx->g_rect[rect_id].left =  atof(x.c_str()) * ctx->dec_width;
                ctx->g_rect[rect_id].top =  atof(y.c_str()) * ctx->dec_height;
                ctx->g_rect[rect_id].width = atof(w.c_str()) * ctx->dec_width;
                ctx->g_rect[rect_id].height = atof(h.c_str()) * ctx->dec_height;

                if (log_level >= LOG_LEVEL_DEBUG)
                {
                    cout << "xywh: " << ctx->g_rect[rect_id].left << ":"
                                     << ctx->g_rect[rect_id].top << ":"
                                     << ctx->g_rect[rect_id].width << ":"
                                     << ctx->g_rect[rect_id].height << endl;
                }

                if (ctx->g_rect[rect_id].width < 10 ||
                    ctx->g_rect[rect_id].height < 10)
                {
                    if (log_level >= LOG_LEVEL_WARN)
                    {
                        cout << "Invalid xywh." << endl;
                    }
                    ctx->g_rect_num--;
                    continue;
                }

                ctx->g_rect[rect_id].border_width = BORDER_WIDTH;
                ctx->g_rect[rect_id].border_color.red = (classNum == 0) ? 1.0f : 0.0;
                ctx->g_rect[rect_id].border_color.green = (classNum == 1) ? 1.0f : 0.0;
                ctx->g_rect[rect_id].border_color.blue = (classNum == 2) ? 1.0f : 0.0;
                ctx->g_rect[rect_id].border_color.alpha = 1.0f;

                rect_id ++;
            }
            else
            {
                if (classNum == FIRST_CLASS_CNT - 1)
                {
                    break;
                }
            }
        }
    }
}

/**
  * Query and Set Capture plane.
  */
static void
query_and_set_capture(context_t * ctx)
{
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    int32_t min_dec_capture_buffers;
    int ret = 0;
    int error = 0;
    uint32_t window_width;
    uint32_t window_height;
    NvBufSurf::NvCommonAllocateParams params;

    /* Get capture plane format from the decoder.
       This may change after resolution change event.
       Refer ioctl VIDIOC_G_FMT */
    ret = dec->capture_plane.getFormat(format);
    TEST_ERROR(ret < 0,
               "Error: Could not get format from decoder capture plane", error);

    /* Get the display resolution from the decoder.
       Refer ioctl VIDIOC_G_CROP */
    ret = dec->capture_plane.getCrop(crop);
    TEST_ERROR(ret < 0,
               "Error: Could not get crop from decoder capture plane", error);

    cout << "Video Resolution: " << crop.c.width << "x" << crop.c.height
        << endl;

    ctx->dec_width = crop.c.width;
    ctx->dec_height = crop.c.height;

    if(ctx->dst_dma_fd != -1)
    {
        ret = NvBufSurf::NvDestroy(ctx->dst_dma_fd);
        ctx->dst_dma_fd = -1;
        TEST_ERROR(ret < 0, "Error: Error in BufferDestroy", error);
    }

    /* Create PitchLinear output buffer for transform. */
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
    params.width = crop.c.width;
    params.height = crop.c.height;
    params.layout = NVBUF_LAYOUT_PITCH;
    if (ctx->out_pixfmt == 1)
      params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    else if (ctx->out_pixfmt == 2)
      params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
    else if (ctx->out_pixfmt == 3)
      params.colorFormat = NVBUF_COLOR_FORMAT_NV16;
    else if (ctx->out_pixfmt == 4)
      params.colorFormat = NVBUF_COLOR_FORMAT_NV24;

    if (ctx->enable_osd_text)
      params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;

    params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

    ret = NvBufSurf::NvAllocate(&params, 1, &ctx->dst_dma_fd);
    TEST_ERROR(ret == -1, "create dmabuf failed", error);

    if (!ctx->disable_rendering)
    {
        /* Destroy the old instance of renderer as resolution
           might have changed */
        delete ctx->renderer;

        if (ctx->fullscreen)
        {
            /* Required for fullscreen */
            window_width = window_height = 0;
        }
        else if (ctx->window_width && ctx->window_height)
        {
            /* As specified by user on commandline */
            window_width = ctx->window_width;
            window_height = ctx->window_height;
        }
        else
        {
            /* Resolution got from the decoder */
            window_width = crop.c.width;
            window_height = crop.c.height;
        }

        /* If height or width are set to zero, EglRenderer creates a fullscreen
           window for rendering */
        ctx->renderer =
            NvEglRenderer::createEglRenderer("renderer0", window_width,
                                           window_height, ctx->window_x,
                                           ctx->window_y);
        TEST_ERROR(!ctx->renderer,
                   "Error in setting up renderer. "
                   "Check if X is running or run with --disable-rendering",
                   error);

        /* Set fps for rendering */
        ctx->renderer->setFPS(ctx->fps);
    }

    /* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
    dec->capture_plane.deinitPlane();

    /* Not necessary to call VIDIOC_S_FMT on decoder capture plane. But
       decoder setCapturePlaneFormat function updates the class variables */
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", error);

    /* Get the min buffers which have to be requested on the capture plane */
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    TEST_ERROR(ret < 0,
               "Error while getting value of minimum capture plane buffers",
               error);

    /* Request, Query and export (min + 5) decoder capture plane buffers.
       Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
    ret =
        dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                       min_dec_capture_buffers + 5, false,
                                       false);
    TEST_ERROR(ret < 0, "Error in decoder capture plane setup", error);

    /* For file write, first deinitialize output and capture planes
       of video converter and then use the new resolution from
       decoder resolution change event */

    /* Start streaming on decoder capture_plane */
    ret = dec->capture_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", error);

    /* Enqueue all the empty capture plane buffers */
    for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
        TEST_ERROR(ret < 0, "Error Qing buffer at output plane", error);
    }

    cout << "Query and set capture successful" << endl;
    return;

error:
    if (error)
    {
        abort(ctx);
        cerr << "Error in " << __func__ << endl;
    }
}

/**
  * Decoder capture thread loop function.
  */
static void *
dec_capture_loop_fcn(void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_event ev;
    int ret;

    cout << "Starting decoder capture loop thread" << endl;
    prctl (PR_SET_NAME, "dec_cap", 0, 0, 0);

    /* Wait for the first Resolution change event as decoder needs
       to know the stream resolution for allocating appropriate
       buffers when calling REQBUFS */
    do
    {
        /* VIDIOC_DQEVENT, max_wait_ms = 1000ms */
        ret = dec->dqEvent(ev, 1000);
        if (ret < 0)
        {
            if (errno == EAGAIN)
            {
                cerr <<
                    "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE"
                    << endl;
            }
            else
            {
                cerr << "Error in dequeueing decoder event" << endl;
            }
            abort(ctx);
            break;
        }
    }
    while (ev.type != V4L2_EVENT_RESOLUTION_CHANGE);

    /* Received the resolution change event, now can do query_and_set_capture */
    if (!ctx->got_error)
        query_and_set_capture(ctx);

    /* Exit on error or EOS which is signalled in main() */
    while (!(ctx->got_error || dec->isInError() || ctx->got_eos))
    {
        NvBuffer *dec_buffer;

        /* Check for resolution change again */
        ret = dec->dqEvent(ev, false);
        if (ret == 0)
        {
            switch (ev.type)
            {
                case V4L2_EVENT_RESOLUTION_CHANGE:
                    query_and_set_capture(ctx);
                    continue;
            }
        }

         /* Decoder capture loop */
        while (1)
        {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;

            /* Dequeue a valid capture_plane buffer that contains YUV BL data */
            if (dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0))
            {
                if (errno == EAGAIN)
                {
                    usleep(1000);
                }
                else
                {
                    abort(ctx);
                    cerr << "Error while calling dequeue at capture plane" <<
                        endl;
                }
                break;
            }

            /* Clip & Stitch can be done by adjusting rectangle. */
            NvBufSurf::NvCommonTransformParams transform_params;
            transform_params.src_top = 0;
            transform_params.src_left = 0;
            transform_params.src_width = ctx->dec_width;
            transform_params.src_height = ctx->dec_height;
            transform_params.dst_top = 0;
            transform_params.dst_left = 0;
            transform_params.dst_width = ctx->dec_width;
            transform_params.dst_height = ctx->dec_height;
            transform_params.flag = NVBUFSURF_TRANSFORM_FILTER;
            transform_params.flip = NvBufSurfTransform_None;
            transform_params.filter = NvBufSurfTransformInter_Nearest;

            /* Perform Blocklinear to PitchLinear conversion. */
            ret = NvBufSurf::NvTransform(&transform_params, dec_buffer->planes[0].fd, ctx->dst_dma_fd);
            if (ret == -1)
            {
                cerr << "Transform failed" << endl;
                break;
            }
            /* Write raw video frame to file. */
            if (ctx->out_file)
            {
                /* Dumping two planes for NV12, NV16, NV24 and three for I420 */
                dump_dmabuf(ctx->dst_dma_fd, 0, ctx->out_file);
                dump_dmabuf(ctx->dst_dma_fd, 1, ctx->out_file);
                if (ctx->out_pixfmt == 2)
                {
                    dump_dmabuf(ctx->dst_dma_fd, 2, ctx->out_file);
                }
            }

            /* Get EGLImage from dmabuf fd */
            NvBufSurface *nvbuf_surf = 0;
            ret = NvBufSurfaceFromFd(ctx->dst_dma_fd, (void**)(&nvbuf_surf));
            if (ret != 0)
            {
                cerr << "Unable to extract NvBufSurfaceFromFd" << endl;
                break;
            }
            if (nvbuf_surf->surfaceList[0].mappedAddr.eglImage == NULL)
            {
                if (NvBufSurfaceMapEglImage(nvbuf_surf, 0) != 0)
                {
                    cerr << "Unable to map EGL Image" << endl;
                    break;
                }
            }

            ctx->egl_image = nvbuf_surf->surfaceList[0].mappedAddr.eglImage;
            if (ctx->egl_image == NULL)
            {
                fprintf(stderr, "Error while mapping dmabuf fd (0x%X) to EGLImage\n",
                         ctx->dst_dma_fd);
                break;
            }

            /* Map EGLImage to CUDA buffer, and call CUDA kernel to
               draw a 32x32 pixels black box on left-top of each frame */
            HandleEGLImage(&ctx->egl_image);

            /* Destroy EGLImage */
            ret = NvBufSurfaceFromFd(ctx->dst_dma_fd, (void**)(&nvbuf_surf));
            if (ret != 0)
            {
                cerr << "Unable to extract NvBufSurfaceFromFd" << endl;
                break;
            }
            if (NvBufSurfaceUnMapEglImage(nvbuf_surf, 0) != 0)
            {
                cerr << "Unable to unmap EGL Image" << endl;
                break;
            }
            ctx->egl_image = NULL;

            if (ctx->enable_osd) {
                get_rect(ctx);
            }

            /* Draw text OSD */
            if (ctx->enable_osd_text) {
                set_text(ctx);
                nvosd_put_text(ctx->nvosd_context,
                                      MODE_CPU,
                                      ctx->dst_dma_fd,
                                      1,
                                      &ctx->textParams);
            }

            /* Draw rectangle OSD */
            if (ctx->g_rect_num > 0) {
                nvosd_draw_rectangles(ctx->nvosd_context,
                                      MODE_GPU,
                                      ctx->dst_dma_fd,
                                      ctx->g_rect_num,
                                      ctx->g_rect);
                nvosd_gpu_apply(ctx->nvosd_context, ctx->dst_dma_fd);
             }

            /* Render converted frame */
            if (!ctx->disable_rendering)
            {
                ctx->renderer->render(ctx->dst_dma_fd);
            }


            /* If not writing to file, Queue the buffer back once it has been used. */
            if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0)
            {
                abort(ctx);
                cerr <<
                    "Error while queueing buffer at decoder capture plane"
                    << endl;
                break;
            }
        }
    }

    cout << "Exiting decoder capture loop thread" << endl;
    return NULL;
}

/**
 * Set the default values for decoder context members.
 */
static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));
    ctx->fullscreen = false;
    ctx->window_height = 0;
    ctx->window_width = 0;
    ctx->window_x = 0;
    ctx->window_y = 0;
    ctx->out_pixfmt = 1;
    ctx->fps = 30;
    ctx->nvosd_context = NULL;
    ctx->dst_dma_fd = -1;
}

int
main(int argc, char *argv[])
{
    context_t ctx;
    int ret = 0;
    int error = 0;
    uint32_t i;
    bool eos = false;
    char *nalu_parse_buffer = NULL;

    /* Set default values for decoder context members */
    set_defaults(&ctx);

    /* After initialization, this thread will feed encoded data to
       outputPlane, so name this thread "OutputPlane" */
    pthread_setname_np(pthread_self(),"OutputPlane");

    if (parse_csv_args(&ctx, argc, argv))
    {
        fprintf(stderr, "Error parsing commandline arguments\n");
        return -1;
    }

    /* Create OSD context and get the OSD file that contains
       OSD coordinate information */
    if (ctx.enable_osd || ctx.enable_osd_text)
        ctx.nvosd_context = nvosd_create_context();
    if (ctx.enable_osd) {
        cout << "ctx.osd_file_path:" << ctx.osd_file_path << endl;

        ctx.osd_file = new ifstream(ctx.osd_file_path);
        TEST_ERROR(!ctx.osd_file->is_open(), "Error opening osd file", cleanup);
    }

    /* Create and initialize video decoder
       more about decoder, refer to 00_video_decode sample */
    ctx.dec = NvVideoDecoder::createVideoDecoder("dec0");
    TEST_ERROR(!ctx.dec, "Could not create decoder", cleanup);

    /* Subscribe to Resolution change event */
    ret = ctx.dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE",
            cleanup);

    /* Set the max size of the outputPlane buffers, here is
       CHUNK_SIZE, which contains the encoded data in bytes */
    ret = ctx.dec->setOutputPlaneFormat(ctx.decoder_pixfmt, CHUNK_SIZE);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    if (ctx.input_nalu)
    {
        nalu_parse_buffer = new char[CHUNK_SIZE];
        ret = ctx.dec->setFrameInputMode(0);
        TEST_ERROR(ret < 0,
                "Error in decoder setFrameInputMode", cleanup);
    }
    else
    {
        /* Set V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control to false
           so that application can send chunks of encoded data instead of forming
           complete frames. This needs to be done before setting format on the
           output plane */
        ret = ctx.dec->setFrameInputMode(1);
        TEST_ERROR(ret < 0,
                "Error in decoder setFrameInputMode", cleanup);
    }

    /* Request MMAP buffers for writing encoded video data */
    ret = ctx.dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Error while setting up output plane", cleanup);

    /* Open local video file */
    ctx.in_file = new ifstream(ctx.in_file_path);
    TEST_ERROR(!ctx.in_file->is_open(), "Error opening input file", cleanup);

    /* Open the output file to save the decoded data/YUV */
    if (ctx.out_file_path)
    {
        ctx.out_file = new ofstream(ctx.out_file_path);
        TEST_ERROR(!ctx.out_file->is_open(), "Error opening output file",
                cleanup);
    }

    /* Start streaming on decoder output_plane */
    ret = ctx.dec->output_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in output plane stream on", cleanup);

    /* Create another thread to capture the decoded output data,
       name the thread "CapturePlane" */
    pthread_create(&ctx.dec_capture_loop, NULL, dec_capture_loop_fcn, &ctx);
    pthread_setname_np(ctx.dec_capture_loop,"CapturePlane");

    /* Read encoded data and enqueue all the output plane buffers.
       Exit loop in case end of file */
    i = 0;
    while (!eos && !ctx.got_error && !ctx.dec->isInError() &&
            i < ctx.dec->output_plane.getNumBuffers())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        buffer = ctx.dec->output_plane.getNthBuffer(i);
        if (ctx.input_nalu)
        {
            read_decoder_input_nalu(ctx.in_file, buffer, nalu_parse_buffer,
                    CHUNK_SIZE);
        }
        else
        {
            read_decoder_input_chunk(ctx.in_file, buffer);
        }

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        /* It is necessary to queue an empty buffer to signal EOS to the decoder
           i.e. set v4l2_buf.m.planes[0].bytesused = 0 and queue the buffer */
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            eos = true;
            cout << "Input file read complete" << endl;
            break;
        }
        i++;
    }

    /* Since all the output plane buffers have been queued in above loop,
       in this loop, firstly dequeue a empty buffer, then read encoded data
       into this buffer, enqueue it back for decoding at last */
    while (!eos && !ctx.got_error && !ctx.dec->isInError())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
        if (ret < 0)
        {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }

        if (ctx.input_nalu)
        {
            read_decoder_input_nalu(ctx.in_file, buffer, nalu_parse_buffer,
                    CHUNK_SIZE);
        }
        else
        {
            read_decoder_input_chunk(ctx.in_file, buffer);
        }
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            eos = true;
            cout << "Input file read complete" << endl;
            break;
        }
    }

    /* As EOS, dequeue all the output planes */
    while (ctx.dec->output_plane.getNumQueuedBuffers() > 0 &&
           !ctx.got_error && !ctx.dec->isInError())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
        if (ret < 0)
        {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
    }

    /* Mark EOS for the decoder capture thread */
    ctx.got_eos = true;

cleanup:
    if (ctx.dec_capture_loop)
    {
        pthread_join(ctx.dec_capture_loop, NULL);
    }

    if (ctx.dec && ctx.dec->isInError())
    {
        cerr << "Decoder is in error" << endl;
        error = 1;
    }
    if (ctx.got_error)
    {
        error = 1;
    }

    /* The decoder destructor does all the cleanup i.e set streamoff on output
       and capture planes, unmap buffers, tell decoder to deallocate buffer
       (reqbufs ioctl with counnt = 0), and finally call v4l2_close on the fd */
    delete ctx.dec;

    /* Similarly, EglRenderer destructor does all the cleanup */
    delete ctx.renderer;
    delete ctx.in_file;
    delete ctx.out_file;

    if(ctx.dst_dma_fd != -1)
    {
        ret = NvBufSurf::NvDestroy(ctx.dst_dma_fd);
        ctx.dst_dma_fd = -1;
        if(ret < 0)
        {
            cerr << "Error in BufferDestroy" << endl;
            error = 1;
        }
    }

    delete []nalu_parse_buffer;

    free(ctx.in_file_path);
    free(ctx.out_file_path);
    if (ctx.enable_osd) {
        ctx.osd_file->close();
        free(ctx.osd_file_path);
    }

    if (ctx.enable_osd_text)
        free(ctx.osd_text);

    if (ctx.enable_osd || ctx.enable_osd_text)
    {
        nvosd_destroy_context(ctx.nvosd_context);
        ctx.nvosd_context = NULL;
    }

    if (error)
    {
        cout << "App run failed" << endl;
    }
    else
    {
        cout << "App run was successful" << endl;
    }
    return -error;
}
