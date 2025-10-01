/*
 * RTSP Stream Ingest - Stream Copy with Audio Support
 *
 * Supports both H.264 and HEVC video codecs, plus audio streams
 *
 * Compile:
 * gcc -o ingest_ffmpeg ingest_ffmpeg_simple.c -lavformat -lavcodec -lavutil -O3
 */

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_RETRIES 10
#define RETRY_DELAY_SEC 5
#define SEGMENT_DURATION_SEC 30
#define MAX_CONSECUTIVE_ERRORS 30

static volatile int keep_running = 1;
static char output_dir[512] = "data/videos";

typedef struct {
    AVFormatContext *input_ctx;
    AVFormatContext *output_ctx;
    int video_stream_idx;
    int audio_stream_idx;
    int *stream_mapping;
    int stream_mapping_size;
    char current_output_file[1024];
    time_t segment_start_real_time;
    int64_t video_pts_offset;
    int64_t video_dts_offset;
    int64_t audio_pts_offset;
    int keyframe_found;
} StreamContext;

void signal_handler(int signum) {
    fprintf(stderr, "\nReceived signal %d, shutting down gracefully...\n", signum);
    keep_running = 0;
}

void create_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0755);
    }
}

void get_daily_directory(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    snprintf(buffer, size, "%s/%04d_%02d_%02d",
             output_dir, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    create_directory(buffer);
}

void get_output_filename(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char daily_dir[512];
    get_daily_directory(daily_dir, sizeof(daily_dir));
    snprintf(buffer, size, "%s/%04d%02d%02d_%02d%02d%02d.mp4",
             daily_dir, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
}

static int interrupt_callback(void *ctx) {
    (void)ctx;
    if (!keep_running) {
        return 1;
    }
    return 0;
}

int open_input_stream(StreamContext *sc, const char *input_url) {
    AVDictionary *opts = NULL;
    int ret;

    // RTSP options for stability
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "5000000", 0);
    av_dict_set(&opts, "timeout", "10000000", 0);
    av_dict_set(&opts, "stimeout", "10000000", 0);
    av_dict_set(&opts, "buffer_size", "4194304", 0); // 4MB for high-res

    sc->input_ctx = avformat_alloc_context();
    if (!sc->input_ctx) {
        fprintf(stderr, "Could not allocate input format context\n");
        av_dict_free(&opts);
        return -1;
    }

    sc->input_ctx->interrupt_callback.callback = interrupt_callback;
    sc->input_ctx->interrupt_callback.opaque = sc;

    ret = avformat_open_input(&sc->input_ctx, input_url, NULL, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
        fprintf(stderr, "Could not open input '%s': %s\n", input_url, av_err2str(ret));
        return ret;
    }

    ret = avformat_find_stream_info(sc->input_ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "Could not find stream info: %s\n", av_err2str(ret));
        return ret;
    }

    // Find video stream
    sc->video_stream_idx = av_find_best_stream(sc->input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (sc->video_stream_idx < 0) {
        fprintf(stderr, "Could not find video stream\n");
        return -1;
    }

    // Find audio stream (optional)
    sc->audio_stream_idx = av_find_best_stream(sc->input_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);

    fprintf(stderr, "Opened input stream: %s\n", input_url);

    AVStream *video_stream = sc->input_ctx->streams[sc->video_stream_idx];
    fprintf(stderr, "Video: %s, %dx%d, %.2f fps\n",
            avcodec_get_name(video_stream->codecpar->codec_id),
            video_stream->codecpar->width,
            video_stream->codecpar->height,
            av_q2d(video_stream->r_frame_rate));

    if (sc->audio_stream_idx >= 0) {
        AVStream *audio_stream = sc->input_ctx->streams[sc->audio_stream_idx];
        fprintf(stderr, "Audio: %s, %d Hz, %d channels\n",
                avcodec_get_name(audio_stream->codecpar->codec_id),
                audio_stream->codecpar->sample_rate,
                audio_stream->codecpar->ch_layout.nb_channels);
    } else {
        fprintf(stderr, "No audio stream found\n");
    }

    return 0;
}

int open_output_file(StreamContext *sc) {
    int ret;
    int i;

    get_output_filename(sc->current_output_file, sizeof(sc->current_output_file));

    ret = avformat_alloc_output_context2(&sc->output_ctx, NULL, "mp4", sc->current_output_file);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate output context: %s\n", av_err2str(ret));
        return ret;
    }

    // Create stream mapping
    sc->stream_mapping_size = sc->input_ctx->nb_streams;
    sc->stream_mapping = av_calloc(sc->stream_mapping_size, sizeof(*sc->stream_mapping));
    if (!sc->stream_mapping) {
        fprintf(stderr, "Could not allocate stream mapping\n");
        return AVERROR(ENOMEM);
    }

    // Initialize all to -1 (not mapped)
    for (i = 0; i < sc->stream_mapping_size; i++) {
        sc->stream_mapping[i] = -1;
    }

    int output_stream_index = 0;

    // Map video stream
    AVStream *in_video_stream = sc->input_ctx->streams[sc->video_stream_idx];
    AVStream *out_video_stream = avformat_new_stream(sc->output_ctx, NULL);
    if (!out_video_stream) {
        fprintf(stderr, "Could not allocate output video stream\n");
        return AVERROR_UNKNOWN;
    }

    ret = avcodec_parameters_copy(out_video_stream->codecpar, in_video_stream->codecpar);
    if (ret < 0) {
        fprintf(stderr, "Could not copy video codec parameters: %s\n", av_err2str(ret));
        return ret;
    }

    out_video_stream->codecpar->codec_tag = 0;
    out_video_stream->time_base = in_video_stream->time_base;
    out_video_stream->r_frame_rate = in_video_stream->r_frame_rate;
    out_video_stream->avg_frame_rate = in_video_stream->avg_frame_rate;

    sc->stream_mapping[sc->video_stream_idx] = output_stream_index++;

    // Map audio stream if present
    if (sc->audio_stream_idx >= 0) {
        AVStream *in_audio_stream = sc->input_ctx->streams[sc->audio_stream_idx];
        AVStream *out_audio_stream = avformat_new_stream(sc->output_ctx, NULL);
        if (!out_audio_stream) {
            fprintf(stderr, "Could not allocate output audio stream\n");
            return AVERROR_UNKNOWN;
        }

        ret = avcodec_parameters_copy(out_audio_stream->codecpar, in_audio_stream->codecpar);
        if (ret < 0) {
            fprintf(stderr, "Could not copy audio codec parameters: %s\n", av_err2str(ret));
            return ret;
        }

        out_audio_stream->codecpar->codec_tag = 0;
        out_audio_stream->time_base = in_audio_stream->time_base;

        sc->stream_mapping[sc->audio_stream_idx] = output_stream_index++;
    }

    // Open output file
    if (!(sc->output_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&sc->output_ctx->pb, sc->current_output_file, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "Could not open output file '%s': %s\n",
                    sc->current_output_file, av_err2str(ret));
            return ret;
        }
    }

    // Set MP4 options
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "movflags", "+faststart", 0);

    ret = avformat_write_header(sc->output_ctx, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
        fprintf(stderr, "Could not write output header: %s\n", av_err2str(ret));
        return ret;
    }

    sc->segment_start_real_time = time(NULL);
    sc->video_pts_offset = -1;
    sc->video_dts_offset = -1;
    sc->audio_pts_offset = -1;
    sc->keyframe_found = 0;

    fprintf(stderr, "Started new segment: %s\n", sc->current_output_file);

    return 0;
}

void close_output_file(StreamContext *sc) {
    if (sc->output_ctx) {
        av_write_trailer(sc->output_ctx);

        if (!(sc->output_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&sc->output_ctx->pb);
        }

        avformat_free_context(sc->output_ctx);
        sc->output_ctx = NULL;

        if (sc->stream_mapping) {
            av_freep(&sc->stream_mapping);
            sc->stream_mapping = NULL;
        }

        fprintf(stderr, "Closed segment: %s\n", sc->current_output_file);
    }
}

int should_start_new_segment(StreamContext *sc) {
    time_t now = time(NULL);
    return (now - sc->segment_start_real_time) >= SEGMENT_DURATION_SEC;
}

int process_stream(StreamContext *sc) {
    AVPacket *pkt = av_packet_alloc();
    int ret = 0;
    int consecutive_errors = 0;

    if (!pkt) {
        fprintf(stderr, "Could not allocate packet\n");
        return -1;
    }

    ret = open_output_file(sc);
    if (ret < 0) {
        av_packet_free(&pkt);
        return ret;
    }

    while (keep_running) {
        // Check if we need a new segment
        if (should_start_new_segment(sc)) {
            close_output_file(sc);
            ret = open_output_file(sc);
            if (ret < 0) {
                break;
            }
        }

        ret = av_read_frame(sc->input_ctx, pkt);
        if (ret < 0) {
            if (ret == AVERROR(EAGAIN)) {
                av_usleep(10000);
                continue;
            }
            if (ret == AVERROR_EOF) {
                fprintf(stderr, "End of stream\n");
                break;
            }

            consecutive_errors++;
            fprintf(stderr, "Error reading frame (attempt %d/%d): %s\n",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS, av_err2str(ret));

            if (consecutive_errors >= MAX_CONSECUTIVE_ERRORS) {
                fprintf(stderr, "Too many consecutive errors, will reconnect\n");
                ret = -1;
                break;
            }

            av_usleep(100000);
            continue;
        }

        consecutive_errors = 0;

        // Only process mapped streams
        if (pkt->stream_index >= sc->stream_mapping_size ||
            sc->stream_mapping[pkt->stream_index] < 0) {
            av_packet_unref(pkt);
            continue;
        }

        // For video packets, wait for keyframe to start segment
        if (pkt->stream_index == sc->video_stream_idx && !sc->keyframe_found) {
            if (!(pkt->flags & AV_PKT_FLAG_KEY)) {
                av_packet_unref(pkt);
                continue;
            }
            sc->keyframe_found = 1;
            fprintf(stderr, "  Video keyframe found, recording started\n");
        }

        // Get streams
        AVStream *in_stream = sc->input_ctx->streams[pkt->stream_index];
        int out_stream_idx = sc->stream_mapping[pkt->stream_index];
        AVStream *out_stream = sc->output_ctx->streams[out_stream_idx];

        // Handle timestamp offsets per stream
        int64_t *pts_offset = NULL;
        int64_t *dts_offset = NULL;

        if (pkt->stream_index == sc->video_stream_idx) {
            pts_offset = &sc->video_pts_offset;
            dts_offset = &sc->video_dts_offset;
        } else if (pkt->stream_index == sc->audio_stream_idx) {
            pts_offset = &sc->audio_pts_offset;
            dts_offset = &sc->audio_pts_offset; // Audio uses same offset
        }

        // Set timestamp offset on first packet of each stream
        if (pts_offset && *pts_offset == -1 && pkt->pts != AV_NOPTS_VALUE) {
            *pts_offset = pkt->pts;
        }
        if (dts_offset && *dts_offset == -1 && pkt->dts != AV_NOPTS_VALUE) {
            *dts_offset = pkt->dts;
        }

        // Adjust timestamps
        if (pkt->pts != AV_NOPTS_VALUE && pts_offset && *pts_offset >= 0) {
            pkt->pts = av_rescale_q(pkt->pts - *pts_offset, in_stream->time_base, out_stream->time_base);
        } else {
            pkt->pts = AV_NOPTS_VALUE;
        }

        if (pkt->dts != AV_NOPTS_VALUE && dts_offset && *dts_offset >= 0) {
            pkt->dts = av_rescale_q(pkt->dts - *dts_offset, in_stream->time_base, out_stream->time_base);
        } else {
            pkt->dts = AV_NOPTS_VALUE;
        }

        pkt->duration = av_rescale_q(pkt->duration, in_stream->time_base, out_stream->time_base);
        pkt->pos = -1;
        pkt->stream_index = out_stream_idx;

        // Write packet
        ret = av_interleaved_write_frame(sc->output_ctx, pkt);
        if (ret < 0) {
            fprintf(stderr, "Error writing packet: %s\n", av_err2str(ret));
        }

        av_packet_unref(pkt);
    }

    close_output_file(sc);
    av_packet_free(&pkt);

    return ret;
}

void cleanup_stream_context(StreamContext *sc) {
    if (sc->input_ctx) {
        avformat_close_input(&sc->input_ctx);
    }
    close_output_file(sc);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_url> [output_dir]\n", argv[0]);
        fprintf(stderr, "Example: %s rtsp://user:pass@192.168.1.100:554/stream data/videos\n", argv[0]);
        return 1;
    }

    const char *input_url = argv[1];
    if (argc >= 3) {
        strncpy(output_dir, argv[2], sizeof(output_dir) - 1);
    }

    create_directory(output_dir);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    fprintf(stderr, "==============================================\n");
    fprintf(stderr, "RTSP Stream Ingest (Video + Audio)\n");
    fprintf(stderr, "==============================================\n");
    fprintf(stderr, "Input: %s\n", input_url);
    fprintf(stderr, "Output: %s\n", output_dir);
    fprintf(stderr, "Segment duration: %d seconds\n", SEGMENT_DURATION_SEC);
    fprintf(stderr, "==============================================\n\n");

    int retry_count = 0;

    while (keep_running && retry_count < MAX_RETRIES) {
        StreamContext sc = {0};
        sc.audio_stream_idx = -1; // Initialize
        int ret;

        ret = open_input_stream(&sc, input_url);
        if (ret < 0) {
            cleanup_stream_context(&sc);
            retry_count++;
            fprintf(stderr, "Retrying in %d seconds... (%d/%d)\n",
                    RETRY_DELAY_SEC, retry_count, MAX_RETRIES);
            sleep(RETRY_DELAY_SEC);
            continue;
        }

        retry_count = 0;
        ret = process_stream(&sc);

        cleanup_stream_context(&sc);

        if (ret < 0 && keep_running) {
            retry_count++;
            fprintf(stderr, "Stream processing failed, retrying in %d seconds... (%d/%d)\n",
                    RETRY_DELAY_SEC, retry_count, MAX_RETRIES);
            sleep(RETRY_DELAY_SEC);
        }
    }

    if (retry_count >= MAX_RETRIES) {
        fprintf(stderr, "Max retries reached, exiting\n");
        return 1;
    }

    fprintf(stderr, "Shutdown complete\n");
    return 0;
}
