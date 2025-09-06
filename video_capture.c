#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <sys/wait.h>

// Platform detection
#ifdef __linux__
    #include <sys/ioctl.h>
    #include <linux/videodev2.h>
    #define PLATFORM_LINUX 1
#elif __APPLE__
    #include <AvailabilityMacros.h>
    #define PLATFORM_MACOS 1
#else
    #error "Unsupported platform"
#endif

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT * 3) // RGB format
#define MAX_BUFFERS 4
#define FRAMES_PER_MINUTE 1800  // 30 FPS * 60 seconds

// Shared memory structure for frame processing
typedef struct {
    int width;
    int height;
    int format;
    size_t size;
    unsigned char data[FRAME_SIZE];
    struct timeval timestamp;
    int frame_ready;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} shared_frame_t;

#ifdef PLATFORM_LINUX
typedef struct {
    void *start;
    size_t length;
} buffer_t;

// Linux V4L2 specific globals
static int fd = -1;
static buffer_t *buffers;
static unsigned int n_buffers;
#endif

#ifdef PLATFORM_MACOS
// macOS specific globals
static FILE *camera_pipe = NULL;
static pthread_t capture_thread;
static unsigned char temp_frame[FRAME_SIZE];
#endif

// Common globals
static int running = 1;
static shared_frame_t *shared_frame = NULL;
static int shm_fd = -1;
static FILE *current_video_pipe = NULL;
static char current_video_path[512];
static time_t current_minute = 0;
static int frame_count_in_minute = 0;

// Function prototypes
static void signal_handler(int sig);
static int init_shared_memory(void);
static void cleanup_shared_memory(void);
static void create_directory_structure(char *path, size_t path_size);
static void open_new_video_file(void);
static void close_current_video(void);
static int init_camera(const char *dev_name);
static int start_capture(void);
static int stop_capture(void);
static void cleanup_camera(void);
static int capture_frame(unsigned char *frame_data, struct timeval *timestamp);
static int check_ffmpeg_availability(void);

#ifdef PLATFORM_LINUX
// Linux V4L2 implementations
static int xioctl(int fh, int request, void *arg);
static int init_device(const char *dev_name);
static int init_mmap(void);
static int start_capturing(void);
static int stop_capturing(void);
static void uninit_device(void);
static void close_device(void);
#endif

#ifdef PLATFORM_MACOS
// macOS implementations
static void* camera_thread_func(void* arg);
static int init_macos_camera(void);
static void cleanup_macos_camera(void);
#endif

static void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    running = 0;
}

static int check_ffmpeg_availability(void) {
    if (system("which ffmpeg > /dev/null 2>&1") != 0) {
        fprintf(stderr, "Error: ffmpeg is required but not found\n");
#ifdef PLATFORM_MACOS
        fprintf(stderr, "Install with: brew install ffmpeg\n");
#else
        fprintf(stderr, "Install with: sudo apt install ffmpeg (Ubuntu) or equivalent\n");
#endif
        return -1;
    }
    
    // Check for H.264 encoder support
    if (system("ffmpeg -encoders 2>/dev/null | grep libx264 > /dev/null") != 0) {
        fprintf(stderr, "Warning: libx264 encoder not found\n");
        return -1;
    }
    
    return 0;
}

static void create_directory_structure(char *path, size_t path_size) {
    time_t now;
    struct tm *tm_info;
    char date_str[32], hour_str[8];
    
    time(&now);
    tm_info = localtime(&now);
    
    strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm_info);
    strftime(hour_str, sizeof(hour_str), "%H", tm_info);
    
    mkdir("videos", 0755);
    
    char date_path[256];
    snprintf(date_path, sizeof(date_path), "videos/%s", date_str);
    mkdir(date_path, 0755);
    
    snprintf(path, path_size, "%s/%s", date_path, hour_str);
    mkdir(path, 0755);
}

static void open_new_video_file(void) {
    time_t now;
    struct tm *tm_info;
    char dir_path[512];
    char minute_str[8];
    char ffmpeg_cmd[1024];
    
    time(&now);
    tm_info = localtime(&now);
    current_minute = now / 60;
    frame_count_in_minute = 0;
    
    create_directory_structure(dir_path, sizeof(dir_path));
    strftime(minute_str, sizeof(minute_str), "%M", tm_info);
    
    snprintf(current_video_path, sizeof(current_video_path), 
             "%s/%s.mp4", dir_path, minute_str);
    
    // Build ffmpeg command for H.264 encoding - simplified for reliability
    snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
        "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r 30 -i - "
        "-c:v libx264 -preset fast -crf 23 "
        "-pix_fmt yuv420p "
        "-movflags +faststart \"%s\"",
        FRAME_WIDTH, FRAME_HEIGHT, current_video_path);
    
    current_video_pipe = popen(ffmpeg_cmd, "w");
    if (!current_video_pipe) {
        fprintf(stderr, "Cannot open ffmpeg pipe for %s: %s\n", 
                current_video_path, strerror(errno));
        return;
    }
    
    printf("Started recording H.264 to: %s\n", current_video_path);
}

static void close_current_video(void) {
    if (current_video_pipe) {
        pclose(current_video_pipe);
        current_video_pipe = NULL;
        printf("Closed video file: %s (frames: %d)\n", current_video_path, frame_count_in_minute);
    }
}

static int init_shared_memory(void) {
    shm_fd = shm_open("/video_capture_frame", O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return -1;
    }
    
    if (ftruncate(shm_fd, sizeof(shared_frame_t)) == -1) {
        perror("ftruncate");
        return -1;
    }
    
    shared_frame = mmap(NULL, sizeof(shared_frame_t), 
                       PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_frame == MAP_FAILED) {
        perror("mmap");
        return -1;
    }
    
    shared_frame->width = FRAME_WIDTH;
    shared_frame->height = FRAME_HEIGHT;
    shared_frame->format = 'RGB ';
    shared_frame->size = FRAME_SIZE;
    shared_frame->frame_ready = 0;
    
    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&shared_frame->mutex, &mutex_attr);
    pthread_mutexattr_destroy(&mutex_attr);
    
    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&shared_frame->cond, &cond_attr);
    pthread_condattr_destroy(&cond_attr);
    
    printf("Shared memory initialized for frame processing\n");
    return 0;
}

static void cleanup_shared_memory(void) {
    if (shared_frame != NULL) {
        pthread_mutex_destroy(&shared_frame->mutex);
        pthread_cond_destroy(&shared_frame->cond);
        munmap(shared_frame, sizeof(shared_frame_t));
        shared_frame = NULL;
    }
    
    if (shm_fd != -1) {
        close(shm_fd);
        shm_unlink("/video_capture_frame");
        shm_fd = -1;
    }
}

#ifdef PLATFORM_LINUX
static int xioctl(int fh, int request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

static int init_device(const char *dev_name) {
    struct v4l2_capability cap;
    struct v4l2_format fmt;

    fd = open(dev_name, O_RDWR | O_NONBLOCK, 0);
    if (-1 == fd) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                dev_name, errno, strerror(errno));
        return -1;
    }

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\n", dev_name);
            return -1;
        } else {
            perror("VIDIOC_QUERYCAP");
            return -1;
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n", dev_name);
        return -1;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n", dev_name);
        return -1;
    }

    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = FRAME_WIDTH;
    fmt.fmt.pix.height = FRAME_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt)) {
        perror("VIDIOC_S_FMT");
        return -1;
    }

    return init_mmap();
}

static int init_mmap(void) {
    struct v4l2_requestbuffers req;

    CLEAR(req);
    req.count = MAX_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf(stderr, "Device does not support memory mapping\n");
            return -1;
        } else {
            perror("VIDIOC_REQBUFS");
            return -1;
        }
    }

    if (req.count < 2) {
        fprintf(stderr, "Insufficient buffer memory\n");
        return -1;
    }

    buffers = calloc(req.count, sizeof(*buffers));
    if (!buffers) {
        fprintf(stderr, "Out of memory\n");
        return -1;
    }

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;

        if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf)) {
            perror("VIDIOC_QUERYBUF");
            return -1;
        }

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start = mmap(NULL, buf.length,
                                       PROT_READ | PROT_WRITE,
                                       MAP_SHARED, fd, buf.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start) {
            perror("mmap");
            return -1;
        }
    }

    return 0;
}

static int start_capturing(void) {
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
            perror("VIDIOC_QBUF");
            return -1;
        }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type)) {
        perror("VIDIOC_STREAMON");
        return -1;
    }

    return 0;
}

static int capture_frame(unsigned char *frame_data, struct timeval *timestamp) {
    struct v4l2_buffer buf;
    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    tv.tv_sec = 2;
    tv.tv_usec = 0;

    r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (-1 == r) {
        if (EINTR == errno)
            return 0;
        perror("select");
        return -1;
    }

    if (0 == r) {
        fprintf(stderr, "select timeout\n");
        return -1;
    }

    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
        switch (errno) {
        case EAGAIN:
            return 0;
        case EIO:
        default:
            perror("VIDIOC_DQBUF");
            return -1;
        }
    }

    memcpy(frame_data, buffers[buf.index].start, 
           buf.bytesused < FRAME_SIZE ? buf.bytesused : FRAME_SIZE);
    *timestamp = buf.timestamp;

    if (-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
        perror("VIDIOC_QBUF");
        return -1;
    }

    return 1;
}

static int stop_capturing(void) {
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type)) {
        perror("VIDIOC_STREAMOFF");
        return -1;
    }
    return 0;
}

static void uninit_device(void) {
    unsigned int i;
    for (i = 0; i < n_buffers; ++i)
        if (-1 == munmap(buffers[i].start, buffers[i].length))
            perror("munmap");
    free(buffers);
}

static void close_device(void) {
    if (-1 == close(fd))
        perror("close");
    fd = -1;
}

static int init_camera(const char *dev_name) {
    return init_device(dev_name);
}

static int start_capture(void) {
    return start_capturing();
}

static int stop_capture(void) {
    return stop_capturing();
}

static void cleanup_camera(void) {
    uninit_device();
    close_device();
}

#endif // PLATFORM_LINUX

#ifdef PLATFORM_MACOS

static void* camera_thread_func(void* arg) {
    (void)arg; // Mark as unused
    char cmd[512];
    snprintf(cmd, sizeof(cmd), 
        "ffmpeg -f avfoundation -video_size %dx%d -framerate 30 -i \"0\" "
        "-pix_fmt rgb24 -f rawvideo - 2>/dev/null", 
        FRAME_WIDTH, FRAME_HEIGHT);
    
    camera_pipe = popen(cmd, "r");
    if (!camera_pipe) {
        fprintf(stderr, "Failed to open camera with ffmpeg\n");
        return NULL;
    }
    
    printf("macOS camera thread started\n");
    
    while (running) {
        size_t bytes_read = fread(temp_frame, 1, FRAME_SIZE, camera_pipe);
        if (bytes_read == FRAME_SIZE) {
            if (shared_frame) {
                gettimeofday(&shared_frame->timestamp, NULL);
            }
            // Frame is ready in temp_frame
        } else if (bytes_read == 0) {
            if (running) {
                fprintf(stderr, "Camera stream ended unexpectedly\n");
            }
            break;
        } else {
            fprintf(stderr, "Partial frame read: %zu bytes\n", bytes_read);
        }
        // No additional delay - let ffmpeg control the frame rate
    }
    
    printf("macOS camera thread ended\n");
    return NULL;
}

static int init_macos_camera(void) {
    return 0;
}

static int capture_frame(unsigned char *frame_data, struct timeval *timestamp) {
    memcpy(frame_data, temp_frame, FRAME_SIZE);
    *timestamp = shared_frame->timestamp;
    return 1;
}

static void cleanup_macos_camera(void) {
    if (camera_pipe) {
        pclose(camera_pipe);
        camera_pipe = NULL;
    }
}

static int init_camera(const char *dev_name) {
    (void)dev_name; // Unused on macOS
    return init_macos_camera();
}

static int start_capture(void) {
    if (pthread_create(&capture_thread, NULL, camera_thread_func, NULL) != 0) {
        perror("pthread_create");
        return -1;
    }
    return 0;
}

static int stop_capture(void) {
    if (pthread_join(capture_thread, NULL) != 0) {
        perror("pthread_join");
        return -1;
    }
    return 0;
}

static void cleanup_camera(void) {
    cleanup_macos_camera();
}

#endif // PLATFORM_MACOS

int main(int argc, char **argv) {
    (void)argc; (void)argv; // Mark as unused for clean compilation
    
    const char *dev_name = NULL;
    
#ifdef PLATFORM_LINUX
    dev_name = argc > 1 ? argv[1] : "/dev/video0";
#elif PLATFORM_MACOS
    dev_name = "0"; // Default camera index for macOS
#endif

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("H.264 Video Capture System Starting...\n");
#ifdef PLATFORM_LINUX
    printf("Platform: Linux (V4L2)\n");
    printf("Device: %s\n", dev_name);
#elif PLATFORM_MACOS
    printf("Platform: macOS (AVFoundation via ffmpeg)\n");
    printf("Camera: %s\n", dev_name);
#endif
    printf("Resolution: %dx%d @ 30fps\n", FRAME_WIDTH, FRAME_HEIGHT);
    printf("Codec: H.264/AVC\n");
    printf("Video files will be saved in videos/YYYY-MM-DD/HH/MM.mp4\n");
    printf("Press Ctrl+C to stop\n\n");

    // Check ffmpeg availability
    if (check_ffmpeg_availability() != 0) {
        return EXIT_FAILURE;
    }

    if (init_shared_memory() != 0) {
        fprintf(stderr, "Failed to initialize shared memory\n");
        return EXIT_FAILURE;
    }

    if (init_camera(dev_name) != 0) {
        cleanup_shared_memory();
        return EXIT_FAILURE;
    }

    if (start_capture() != 0) {
        cleanup_camera();
        cleanup_shared_memory();
        return EXIT_FAILURE;
    }

    open_new_video_file();

    // Main capture loop
    unsigned char frame_buffer[FRAME_SIZE];
    while (running) {
        struct timeval timestamp;
        int frame_result = capture_frame(frame_buffer, &timestamp);
        
        if (frame_result > 0) {
            // Check if we need to start a new video file
            time_t now = time(NULL);
            if (now / 60 != current_minute) {
                close_current_video();
                open_new_video_file();
            }

            // Write frame to ffmpeg pipe for H.264 encoding
            if (current_video_pipe) {
                if (fwrite(frame_buffer, FRAME_SIZE, 1, current_video_pipe) != 1) {
                    fprintf(stderr, "Failed to write frame to ffmpeg pipe\n");
                } else {
                    frame_count_in_minute++;
                    fflush(current_video_pipe);
                }
            }

            // Update shared memory for frame processing
            if (shared_frame) {
                pthread_mutex_lock(&shared_frame->mutex);
                memcpy(shared_frame->data, frame_buffer, FRAME_SIZE);
                shared_frame->timestamp = timestamp;
                shared_frame->frame_ready = 1;
                pthread_cond_signal(&shared_frame->cond);
                pthread_mutex_unlock(&shared_frame->mutex);
            }
        } else if (frame_result < 0) {
            fprintf(stderr, "Frame capture error, continuing...\n");
            usleep(100000); // 100ms delay on error
        }
    }

    printf("\nShutting down...\n");
    
    close_current_video();
    stop_capture();
    cleanup_camera();
    cleanup_shared_memory();

    printf("H.264 video capture stopped cleanly\n");
    return EXIT_SUCCESS;
}