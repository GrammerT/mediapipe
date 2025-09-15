#include "person_camera_detection_module.h"
#include <iostream>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "internal_def.h"
#ifdef WIN32
    #include "windows.h"
#endif

#include <fstream>

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

constexpr char kFaceDetections[] = "face_detections";
constexpr char kObjectDetections[] = "output_detections";

constexpr char kOriginWindow[] = "OriginWindow";
constexpr char kFaceDetection[] = "FaceDetection";
constexpr char kObjectDetection[] = "ObjectDetection";

#define CAMERA_DETECTION_TIMER 700 // 0.7s


// 自定义streambuf，自动为每行日志添加时间戳
class TimePrefixBuf : public std::streambuf {
public:
    TimePrefixBuf(std::streambuf* dest) : dest_(dest), at_line_start_(true) {}
protected:
    virtual int overflow(int c) override {
        if (c == traits_type::eof()) return traits_type::not_eof(c);
        if (at_line_start_) {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm;
            #ifdef _WIN32
                localtime_s(&tm, &t);
            #else
                localtime_r(&t, &tm);
            #endif
            char buf[32];
            std::strftime(buf, sizeof(buf), "[%Y-%m-%d %H:%M:%S] ", &tm);
            dest_->sputn(buf, std::strlen(buf));
            at_line_start_ = false;
        }
        if (c == '\n') at_line_start_ = true;
        return dest_->sputc(c);
    }
    std::streambuf* dest_;
    bool at_line_start_;
};

void log_init()
{
    // 获取当前运行目录
    char current_path[1024] = {0};
    #ifdef _WIN32
        GetModuleFileNameA(NULL, current_path, sizeof(current_path));
        char* last_slash = strrchr(current_path, '\\');
        if (last_slash) {
            *last_slash = '\0';
        }
    #else
        ssize_t count = readlink("/proc/self/exe", current_path, sizeof(current_path) - 1);
        if (count != -1) {
            current_path[count] = '\0';
            char* last_slash = strrchr(current_path, '/');
            if (last_slash) {
                *last_slash = '\0';
            }
        }
    #endif
    // 设置日志文件保存目录
    std::string log_dir = std::string(current_path) + "\\Log";
    static std::ofstream log_stream;
    log_stream.open(log_dir + "\\detect_module.log", std::ios::out | std::ios::trunc);
    if (log_stream.is_open()) {
        log_stream.setf(std::ios::unitbuf); // 保证每次输出后立即刷新缓冲区
        static TimePrefixBuf time_buf(log_stream.rdbuf());
        std::cout.rdbuf(&time_buf);
        std::cerr.rdbuf(&time_buf);
    } else {
        std::cerr << "Failed to open log file for stdout/stderr redirection." << std::endl;
    }
}




PersonCameraDetectionModule::PersonCameraDetectionModule(bool create_log) {
    if(create_log)
    {
        log_init();
    }
    // std::cout << "PersonCameraDetectionModule initialized." << std::endl;
}

PersonCameraDetectionModule::~PersonCameraDetectionModule() {
    // Destructor implementation
    std::unique_lock<std::mutex> lock(m_camera_mutex);
    if (m_camera) {
        m_camera->release();
        m_camera.reset();
    }
    lock.unlock();
    StopDetection();
    // std::cout << "PersonCameraDetectionModule destroyed." << std::endl;
    // google::ShutdownGoogleLogging();
}

DetectionError PersonCameraDetectionModule::Initialize(const GeneralConfig& generalConfig, 
    const CrowdDetectionConfig& crowdConfig, 
    const PhotoLeakDetectionConfig& photoLeakConfig, 
    const AbsenceDetectionConfig& absenceConfig) {
    // std::cout << "GeneralConfig details:" << std::endl;
    // std::cout << "Camera Timeout: " << generalConfig.camera.timeout << std::endl;
    // std::cout << "Camera Resolution: " << generalConfig.camera.width << "x" << generalConfig.camera.height << std::endl;
    // std::cout << "Camera Frame Rate: " << generalConfig.camera.frame_rate << std::endl;
    // std::cout << "Debug Mode: " << (generalConfig.debug_mode ? "Enabled" : "Disabled") << std::endl;
    // Deep copy for char* members in GeneralConfig
    // If GeneralConfig contains char* fields, copy their contents instead of pointers
    // Example assumes a field: char* name;
    // Adjust according to your actual struct definition

    // Copy other primitive fields
    m_general_config = generalConfig;

    m_crowd_config = crowdConfig;
    m_photo_leak_config = photoLeakConfig;
    m_absence_config = absenceConfig;

    // Initialize the camera
    DetectionError error = InitializeCamera(generalConfig);
    if (error != DetectionError::None) {
        // std::cout << "Failed to initialize camera. Error code: " << static_cast<int>(error) << std::endl;
        return error;
    }
    // Initialize the detection graphs
    error = InitializeFaceDetectionGraph();
    if (error != DetectionError::None) {
        // std::cout << "Failed to initialize face detection graph." << std::endl;
        return error;
    }

    error = InitializeObjectDetectionGraph(photoLeakConfig);
    if (error != DetectionError::None) {
        // std::cout << "Failed to initialize object detection graph." << std::endl;
        return error;
    }
    m_already_initialized = true;
    // std::cout << "Module initialized with camera timeout: " << generalConfig.camera.timeout
    //           << " and resolution:" << generalConfig.camera.width << "x" << generalConfig.camera.height << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::UpdateGeneralConfig(const GeneralConfig& generalConfig) {
    if (memcmp(&m_general_config, &generalConfig, sizeof(GeneralConfig)) != 0) {
        m_general_config = generalConfig;
        // std::cout << "GeneralConfig updated." << std::endl;
    }
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::UpdateCrowdDetectionConfig(const CrowdDetectionConfig& crowdConfig) {
    if (memcmp(&m_crowd_config, &crowdConfig, sizeof(CrowdDetectionConfig)) != 0) {
        m_crowd_config = crowdConfig;
        // std::cout << "CrowdDetectionConfig updated." << std::endl;
    }
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::UpdatePhotoLeakDetectionConfig(const PhotoLeakDetectionConfig& photoLeakConfig) {
    if (memcmp(&m_photo_leak_config, &photoLeakConfig, sizeof(PhotoLeakDetectionConfig)) != 0) {
        m_photo_leak_config = photoLeakConfig;
        // std::cout << "PhotoLeakDetectionConfig updated." << std::endl;
    }
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::UpdateAbsenceDetectionConfig(const AbsenceDetectionConfig& absenceConfig) {
    if (memcmp(&m_absence_config, &absenceConfig, sizeof(AbsenceDetectionConfig)) != 0) {
        m_absence_config = absenceConfig;
        // std::cout << "AbsenceDetectionConfig updated." << std::endl;
    }
    return DetectionError::None;
}


DetectionError PersonCameraDetectionModule::StartDetection(DetectionResultCallback callback, void* user_data) {
    if (m_detection_running) {
        // std::cout << "Detection is already running." << std::endl;
        return DetectionError::None;
    }

    // std::cout << "ResumeCameraCapture will started ." << std::endl;
    // DetectionError err = ResumeCameraCapture();
    // if (err != DetectionError::None) {
    //     std::cout << "Failed to resume camera capture." << std::endl;
    //     return err;
    // }
    // std::cout << "AddOutputStreamPoller will started ." << std::endl;

    m_face_img_poller = 
        m_face_detection_graph->AddOutputStreamPoller(kOutputStream);
    if (!m_face_img_poller.ok())
    {
        // std::cout << "Failed to add output stream poller for face detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }

    m_face_detection_poller = 
        m_face_detection_graph->AddOutputStreamPoller(kFaceDetections);
    if (!m_face_detection_poller.ok())
    {
        // std::cout << "Failed to add output stream poller for face detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }
    // std::cout << "m_face_poller AddOutputStreamPoller finished ." << std::endl;

    m_object_poller = 
        m_object_detection_graph->AddOutputStreamPoller(kOutputStream);
    if (!m_object_poller.ok())
    {
        // std::cout << "Failed to add output stream poller for object detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }

    m_object_direction_poller = m_object_detection_graph->AddOutputStreamPoller(kObjectDetections);
    if (!m_object_direction_poller.ok())
    {
        // std::cout << "Failed to add objection output stream direction poller for object detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }

    // std::cout << "m_object_poller AddOutputStreamPoller finished ." << std::endl;
    // std::cout << "Detection will started ." << std::endl;
    auto status = m_face_detection_graph->StartRun({});
    if (!status.ok()) {
        // std::cout << "Failed to start face detection graph: " << status.message() << std::endl;
        return DetectionError::MMPStartFailed;
    }
    status = m_object_detection_graph->StartRun({});
    if (!status.ok()) {
        // std::cout << "Failed to start object detection graph: " << status.message() << std::endl;
        return DetectionError::MMPStartFailed;
    }
    // Start the detection thread
    startInterfaceDetectionThread(callback, user_data);
    // std::cout << "Detection thread started successfully." << std::endl;
    m_detection_running = true;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::StopDetection() {
    if (!m_detection_running) {
        // std::cout << "Detection is not running." << std::endl;
        return DetectionError::None;
    }
    m_detection_running = false;
    stopInterfaceDetectionThread();
    std::unique_lock<std::mutex> lock(m_camera_mutex);
    if (m_camera) {
        m_camera->release(); // 释放摄像头资源
        m_camera.reset();
    }
    lock.unlock();
    if (m_face_detection_graph) {
        m_face_detection_graph->CloseInputStream(kInputStream);
        m_face_detection_graph->WaitUntilDone();
        m_face_detection_graph.reset();
    }
    if (m_object_detection_graph) {
        m_object_detection_graph->CloseInputStream(kInputStream);
        m_object_detection_graph->WaitUntilDone();
        m_object_detection_graph.reset();
    }
    // std::cout << "Detection stopped." << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::DetectFromImage(const uint8_t* image_data, int width, int height, 
                                                        DetectionResultCallback callback, void* user_data) {
    if (!image_data || width <= 0 || height <= 0) {
        // std::cout << "Invalid image data." << std::endl;
        return DetectionError::ProcessingError;
    }

    // Simulate image detection
    DetectionResult result;
    result.person_count = 1;  // Example: detected one person
    callback(&result,user_data);

    // std::cout << "Image detection completed for resolution: " << width << "x" << height << std::endl;
    return DetectionError::None;
    }

DetectionError PersonCameraDetectionModule::GetDetectionResult(DetectionResult& result) const {
    if (!m_detection_running) {
        // std::cout << "Detection is not running." << std::endl;
        return DetectionError::Uninitialized;
    }

    // Simulate returning a detection result
    result.person_count = 1;  // Example: detected one person
    result.is_absent = false;
    result.is_photo_leak_possible = false;

    // std::cout << "Returning detection result." << std::endl;
    return DetectionError::None;
}

bool PersonCameraDetectionModule::IsDetectionRunning() const {
    return m_detection_running;
}


DetectionError PersonCameraDetectionModule::InitializeFaceDetectionGraph() {
    // Simulate initialization of the face detection graph
    // std::cout << "Initializing face detection graph..." << std::endl;

    // std::cout << "Initialize the calculator graph." << std::endl;
    std::string calculator_graph_config_contents = generateFaceDetectionGraph();
    mediapipe::CalculatorGraphConfig config =
                            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
                                                        calculator_graph_config_contents);

    m_face_detection_graph = std::make_unique<mediapipe::CalculatorGraph>();
    
    absl::Status status = m_face_detection_graph->Initialize(config);
    if (!status.ok()) {
        // std::cout << "Failed to initialize face detection graph: " << status.message() << std::endl;
        return DetectionError::MMPInitializationFailed;
    }
    // std::cout << "Face detection graph initialized successfully." << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::InitializeObjectDetectionGraph(const PhotoLeakDetectionConfig& photoLeakConfig) {
    // Simulate initialization of the object detection graph
    // std::cout << "Initializing object detection graph..." << std::endl;

    std::string calculator_graph_config_contents = generateObjectionDetectionGraph(photoLeakConfig.pose_threshold,1);
    mediapipe::CalculatorGraphConfig config =
                            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
                                                        calculator_graph_config_contents);

    m_object_detection_graph = std::make_unique<mediapipe::CalculatorGraph>();
    
    absl::Status status = m_object_detection_graph->Initialize(config);
    if (!status.ok()) {
        // std::cout << "Failed to initialize face detection graph: " << status.message() << std::endl;
        return DetectionError::MMPInitializationFailed;
    }

    // std::cout << "Object detection graph initialized successfully." << std::endl;
    return DetectionError::None;
}


DetectionError PersonCameraDetectionModule::InitializeCamera(const GeneralConfig& generalConfig) {
    // std::cout << "Initializing camera with timeout: " << generalConfig.camera.timeout
    //           << " and resolution: " << generalConfig.camera.width << "x" <<
    //           generalConfig.camera.height << std::endl;

    // Open the camera using OpenCV
    // std::unique_lock<std::mutex> lock(m_camera_mutex);
    // m_camera = std::make_unique<cv::VideoCapture>();  // Open default camera (index 0)
    // std::cout << "Camera initialized successfully." << std::endl;
    return DetectionError::None;
}


DetectionError PersonCameraDetectionModule::PauseCameraCapture() {
    std::unique_lock<std::mutex> lock(m_camera_mutex);
    if (!m_camera || !m_camera->isOpened()) {
        // std::cout << "Camera is not initialized or already closed." << std::endl;
        lock.unlock();
        return DetectionError::None;
    }
    
    if (m_camera->isOpened()) {
        // std::cout <<" Pausing camera capture..." << std::endl;
        m_camera->release(); // 释放摄像头资源
        m_camera.reset(); // Reset the camera pointer
        lock.unlock();
        m_camera_opened.store(false);  // Set camera opened flag to false
        // std::cout << "Camera capture paused successfully." << std::endl;
        return DetectionError::None;
    }
    // std::cout << "Camera capture is already paused." << std::endl;
    return DetectionError::InvalidOperation;
}


DetectionError PersonCameraDetectionModule::ResumeCameraCapture() {
    std::unique_lock<std::mutex> lock(m_camera_mutex);
    if (!m_camera) {
        m_camera = std::make_unique<cv::VideoCapture>();
    }
    
    if (!m_camera->isOpened()) {
        // std::cout << "will open camera 0" << std::endl;
        m_camera_detected = false;
        bool camera_opened = false;
        for (int i = 0; i <= 10; ++i) {
          m_camera->open(i, cv::CAP_DSHOW);  // Use DirectShow backend explicitly for better compatibility.
          if (m_camera->isOpened()) {
            // std::cout << "Successfully opened webcam at index " << i << std::endl;
            camera_opened = true;
            break;
          } else {
            // std::cout << "Failed to open webcam at index " << i << std::endl;
          }
        }
        if (!camera_opened) {
        //   std::cout << "Failed to open any webcam. Please check the device connections." << std::endl;
          return DetectionError::CameraStartFailed;
        }

        // std::cout << "opened camera 0" << std::endl;
        if (!m_camera->isOpened()) {
            // std::cout << "Failed to resume camera capture." << std::endl;
            return DetectionError::CameraStartFailed;
        }
        m_camera_opened.store(true);  // Set camera opened flag to true
        // Set camera resolution
        m_camera->set(cv::CAP_PROP_FRAME_WIDTH, m_general_config.camera.width);
        m_camera->set(cv::CAP_PROP_FRAME_HEIGHT, m_general_config.camera.height);
        // Set camera frame rate
        m_camera->set(cv::CAP_PROP_FPS, m_general_config.camera.frame_rate);
        return DetectionError::None;
    }

    // std::cout << "Camera capture is already running." << std::endl;
    return DetectionError::None;
}

void PersonCameraDetectionModule::startInterfaceDetectionThread(DetectionResultCallback callback, void* user_data) {
    if (m_inference_thread.joinable()) {
        // std::cout << "Detection thread is already running." << std::endl;
        return;
    }
    m_detection_running = true;
    if (m_general_config.debug_mode)
    {
        // std::cout << "Debug mode is enabled. Creating debug windows." << std::endl;
        cv::namedWindow(kOriginWindow, /*flags=WINDOW_AUTOSIZE*/ 1);
        cv::namedWindow(kFaceDetection, /*flags=WINDOW_AUTOSIZE*/ 1);
        cv::namedWindow(kObjectDetection, /*flags=WINDOW_AUTOSIZE*/ 1);
    }
    
    m_inference_thread = std::thread([this, callback,user_data]() {
        while (m_detection_running) {
            std::unique_lock<std::mutex> lock(m_camera_mutex);
            if (!m_camera || !m_camera->isOpened()) {
                // std::cout << "Camera is not initialized or closed." << std::endl;
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000/m_general_config.camera.frame_rate));
                continue;
            }
            if (!m_camera_opened.load())
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000/m_general_config.camera.frame_rate));
                continue;
            }
            cv::Mat camera_frame_raw;
            *m_camera>>camera_frame_raw;
            lock.unlock();
            if (camera_frame_raw.empty()) {
                // std::cout << "Ignore empty frames from Queue." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            {
                // std::unique_lock<std::mutex> lock(m_face_img_mutex);
                // camera_frame_raw.copyTo(m_last_mat_face_detect);
            }
            cv::Mat camera_frame;
            cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
            cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
            
            // if (m_general_config.debug_mode) {
            //     cv::imshow(kOriginWindow, camera_frame);
            //     cv::waitKey(5);  // Display the frame for 1 ms
            // }
            // Wrap Mat into an ImageFrame.
            auto face_input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary);
            auto object_input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary);

            cv::Mat face_input_frame_mat = mediapipe::formats::MatView(face_input_frame.get());
            cv::Mat object_input_frame_mat = mediapipe::formats::MatView(object_input_frame.get());

            camera_frame.copyTo(face_input_frame_mat);
            camera_frame.copyTo(object_input_frame_mat);

            size_t frame_timestamp_us =
                (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
            
            auto addRetStatus = m_face_detection_graph->AddPacketToInputStream(
                    kInputStream, mediapipe::Adopt(face_input_frame.release())
                                      .At(mediapipe::Timestamp(frame_timestamp_us)));
            auto addRetStatus2 = m_object_detection_graph->AddPacketToInputStream(
                    kInputStream, mediapipe::Adopt(object_input_frame.release())
                                      .At(mediapipe::Timestamp(frame_timestamp_us)));

            if (!addRetStatus.ok()||!addRetStatus2.ok()) {
                // std::cout << "Failed to send frame to face detection graph: "
                //           << addRetStatus.message() << std::endl;
                continue;
            }
            // std::cout << "Processing frame at timestamp: " << frame_timestamp_us << " us" << std::endl;

            // DetectionResult result;
            m_callback_result.is_camera_blocked = IsCameraPossiblyBlocked(camera_frame);
            processFaceDetectionResult(m_callback_result);
            processObjectDetectionResult(m_callback_result);
            // Output detection result to the log
            // std::cout << "Detection Result: " 
            //           << "Person Count: " << m_callback_result.person_count 
            //           << ", Is Absent: " << m_callback_result.is_absent 
            //           << ", Absence Timeout: " << m_callback_result.absence_timeout 
            //           << ", Is Photo Leak Possible: " << m_callback_result.is_photo_leak_possible 
            //           << ", Is Camera Blocked: " << m_callback_result.is_camera_blocked << std::endl;

            // Invoke the callback with the detection result
            if(callback)
            {
                callback(&m_callback_result,user_data);
            }
            m_callback_result.person_count = 0; // Reset for next frame
            m_callback_result.is_absent = false; // Reset for next frame
            m_callback_result.is_photo_leak_possible = false; // Reset for next frame
            m_callback_result.is_camera_blocked = false; // Reset for next frame
            m_callback_result.absence_timeout = 0; // Reset for next frame
            m_callback_result.photo_leak_confidence = 0.0f;
           // Log detection results for debugging
            // std::cout << "Processed frame at timestamp: " << frame_timestamp_us << " us" << std::endl;            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000/m_general_config.camera.frame_rate));
        }
    });

    // std::cout << "Detection thread started." << std::endl;
}

void PersonCameraDetectionModule::stopInterfaceDetectionThread() {
    m_detection_running = false;
    if (m_inference_thread.joinable()) {
        m_inference_thread.join();
    }
    // std::cout << "Detection thread stopped." << std::endl;
}


void PersonCameraDetectionModule::dealAbsentDetection(DetectionResult& result) {
    if (result.is_absent) { // If absence is detected
        if (!m_already_recode_time) {
            m_already_recode_time = true;
            result.absence_timeout = 0;
            m_absence_start_time = std::chrono::steady_clock::now();
            // std::cout << "Absence detected, starting timer." << std::endl;
        } else {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - m_absence_start_time).count();
            result.absence_timeout = elapsed_time;
            // std::cout << "Absence detected, elapsed time: " << elapsed_time << " seconds." << std::endl;
        }
    } else { // If presence is detected
        if (m_already_recode_time) {
            m_already_recode_time = false;
            // std::cout << "Presence detected, resetting absence timer." << std::endl;
        }
    }
}


void PersonCameraDetectionModule::processFaceDetectionResult(DetectionResult &result)
{
    // Poll face detection results
    mediapipe::Packet face_packet;
    if (m_face_detection_poller->QueueSize() > 0) {
        // std::cout << "m_face_detection_poller->QueueSize() > 0" << std::endl;
        if (m_face_detection_poller->Next(&face_packet)) {
            // Process face detection results
            auto& detections = face_packet.Get<std::vector<mediapipe::Detection>>();
            result.person_count = detections.size();  // Number of faces detected
            result.is_absent = false;  // Example: set to false for now
            dealAbsentDetection(result);
        }
        if(m_face_img_poller->QueueSize() > 0) {
            if (m_face_img_poller->Next(&face_packet)) {
                auto& output_frame = face_packet.Get<mediapipe::ImageFrame>();
                cv::Mat face_detect = mediapipe::formats::MatView(&output_frame);
                cv::cvtColor(face_detect, face_detect, cv::COLOR_RGB2BGR);
                std::unique_lock<std::mutex> lock(m_face_img_mutex);
                face_detect.copyTo(m_last_mat_face_detect);
            }
        }
    } else {
        if(m_face_img_poller->QueueSize() > 0) {
            if (m_face_img_poller->Next(&face_packet)) {
                auto& output_frame = face_packet.Get<mediapipe::ImageFrame>();
                cv::Mat face_detect = mediapipe::formats::MatView(&output_frame);
                cv::cvtColor(face_detect, face_detect, cv::COLOR_RGB2BGR);
                std::unique_lock<std::mutex> lock(m_face_img_mutex);
                face_detect.copyTo(m_last_mat_face_detect);
            }
            result.person_count = 0;
            result.is_absent = true;
            dealAbsentDetection(result);
        }    
    }
}


void PersonCameraDetectionModule::processObjectDetectionResult(DetectionResult &result)
{
    if (m_object_direction_poller->QueueSize() > 0) {
        // std::cout << "m_object_direction_poller->QueueSize() > 0" << std::endl;
        mediapipe::Packet obj_packet;
        if (m_object_direction_poller->Next(&obj_packet)) {
            // Process face detection results
            auto detections = obj_packet.Get<std::vector<mediapipe::Detection>>();
            for (const auto& detection : detections) 
            {
                const auto num_labels =
                    std::max(detection.label_size(), detection.label_id_size());
                if (num_labels > 0)
                {
                    if(!m_camera_detected ) {
                        m_photo_leak_confidences.clear();
                        m_photo_leak_confidences.push_back(detection.score(0));
                        m_camera_detected = true;
                        m_need_save_obj_mat = true;
                        m_detection_start_time = std::chrono::steady_clock::now();
                        // std::cout << "Camera detected, starting timer." << std::endl;
                        break;
                    }
                    else
                    {
                        // std::cout<<"photo leak confidence: "<<detection.score(0)<<std::endl;
                        m_photo_leak_confidences.push_back(detection.score(0));
                        auto current_time = std::chrono::steady_clock::now();
                        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - m_detection_start_time).count();
                        if (elapsed_time > CAMERA_DETECTION_TIMER) {
                            // Camera detected for too long, set to absent
                            result.is_photo_leak_possible = true;  // Example: set to true for now                       
                            // 求m_photo_leak_confidences的平均值
                            float sum_conf = 0.0f;
                            for (float v : m_photo_leak_confidences) {
                                sum_conf += v;
                            }
                            result.photo_leak_confidence = sum_conf / m_photo_leak_confidences.size();
                        }
                    }
                }
            }
            if(detections.size()<=0)
            {
                m_need_save_obj_mat = false;
                m_camera_detected = false;
            }
        }
        
        mediapipe::Packet object_packet;
        if (m_object_poller->Next(&object_packet)) {
           auto& output_frame = object_packet.Get<mediapipe::ImageFrame>();
               // Convert back to opencv for display or saving.
            if(m_need_save_obj_mat)
            {
                m_need_save_obj_mat = false;
                cv::Mat object_detect = mediapipe::formats::MatView(&output_frame);
                cv::cvtColor(object_detect, object_detect, cv::COLOR_RGB2BGR);
                std::unique_lock<std::mutex> lock(m_obj_img_mutex);
                object_detect.copyTo(m_last_mat_object_detect);
            }

        }
    }
    else
    {
        //! 表示没有检测到相机
        m_camera_detected = false;
    } 

    if (m_object_poller->QueueSize() > 0) {
        // Poll object detection results
        mediapipe::Packet object_packet;
        if (m_object_poller->Next(&object_packet)) {
           auto& output_frame = object_packet.Get<mediapipe::ImageFrame>();
            if(m_need_save_obj_mat)
            {
                m_need_save_obj_mat = false;
                cv::Mat object_detect = mediapipe::formats::MatView(&output_frame);
                cv::cvtColor(object_detect, object_detect, cv::COLOR_RGB2BGR);
                std::unique_lock<std::mutex> lock(m_obj_img_mutex);
                object_detect.copyTo(m_last_mat_object_detect);
            }
        }
    } 
}


bool PersonCameraDetectionModule::IsCameraPossiblyBlocked(cv::Mat image) const
{
    if (image.empty()) {
        // std::cout << "Input image is empty." << std::endl;
        return false;
    }
    cv::Mat gray_image;
    image.copyTo(gray_image);
    cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray_image, mean, stddev);

    double variance = stddev[0] * stddev[0];
    if (variance < 200) {
        return true; // Camera is possibly blocked
    } else {
        return false; // Camera is not blocked
    }
}


DetectionError PersonCameraDetectionModule::SaveImage(const char* file_path,bool face_detect)
{
    std::string file_path_str;
#ifdef _WIN32
    // Windows环境下，将UTF-8编码的路径转换为ANSI编码
    int wlen = MultiByteToWideChar(CP_UTF8, 0, file_path, -1, nullptr, 0);
    if (wlen > 0) {
        std::wstring wfile_path(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, file_path, -1, &wfile_path[0], wlen);
        
        int len = WideCharToMultiByte(CP_ACP, 0, wfile_path.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (len > 0) {
            file_path_str.resize(len - 1);
            WideCharToMultiByte(CP_ACP, 0, wfile_path.c_str(), -1, &file_path_str[0], len, nullptr, nullptr);
        } else {
            file_path_str = file_path; // 转换失败时使用原始路径
        }
    } else {
        file_path_str = file_path; // 转换失败时使用原始路径
    }
#else
    // Linux环境下直接使用UTF-8编码
    file_path_str = file_path;
#endif
    if (file_path_str.empty()) {
        // std::cout << "File path is empty." << std::endl;
        return DetectionError::InvalidArgument;
    }
    if(face_detect)
    {
        std::unique_lock<std::mutex> lock(m_face_img_mutex);
        if (m_last_mat_face_detect.empty()) {
            // std::cout << "No face detection image available." << std::endl;
            return DetectionError::ProcessingError;
        }
        if (!cv::imwrite(file_path_str.c_str(), m_last_mat_face_detect)) {
            // std::cout << "Failed to write temporary face detection image." << std::endl;
            return DetectionError::ProcessingError;
        }
        m_last_mat_face_detect.release(); // 释放图像资源
        // std::cout << "Face detection image saved to: " << file_path_str.c_str() << std::endl;
        return DetectionError::None;
    }
    else
    {
        std::unique_lock<std::mutex> lock(m_obj_img_mutex);
        if (m_last_mat_object_detect.empty()) {
            // std::cout << "No object detection image available." << std::endl;
            return DetectionError::ProcessingError;
        }
        if (!cv::imwrite(file_path_str.c_str(), m_last_mat_object_detect)) {
            // std::cout << "Failed to write temporary object detection image." << std::endl;
            return DetectionError::ProcessingError;
        }
        m_last_mat_object_detect.release(); // 释放图像资源
        // std::cout << "Object detection image saved to: " << file_path_str.c_str() << std::endl;
        return DetectionError::None;
    }
    return DetectionError::None;
}

bool PersonCameraDetectionModule::HasCameraDevice() const {
    // Try to open camera devices from index 0 to 10
    for (int i = 0; i <= 10; ++i) {
        cv::VideoCapture temp_camera;
        temp_camera.open(i, cv::CAP_DSHOW);
        if (temp_camera.isOpened()) {
            temp_camera.release();
            return true;
        }
    }
    return false;
}



extern "C" {

PERSON_CAMERA_DETECTION_MODULE_API IPersonCameraDetectionModule* CreatePersonCameraDetectionModule(bool createLog = false) {
    return new PersonCameraDetectionModule(createLog);
}

PERSON_CAMERA_DETECTION_MODULE_API void DestroyPersonCameraDetectionModule(IPersonCameraDetectionModule* module) {
    delete module;
}

};