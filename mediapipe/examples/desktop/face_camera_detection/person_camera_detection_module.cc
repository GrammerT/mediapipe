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

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

constexpr char kFaceDetections[] = "face_detections";
constexpr char kObjectDetections[] = "output_detections";

constexpr char kOriginWindow[] = "OriginWindow";
constexpr char kFaceDetection[] = "FaceDetection";
constexpr char kObjectDetection[] = "ObjectDetection";

#define CAMERA_DETECTION_TIMER 2000 // 2s

PersonCameraDetectionModule::PersonCameraDetectionModule() {
    // Constructor implementation
    std::cout << "PersonCameraDetectionModule initialized." << std::endl;
}

PersonCameraDetectionModule::~PersonCameraDetectionModule() {
    // Destructor implementation
    std::cout << "PersonCameraDetectionModule destroyed." << std::endl;
}

DetectionError PersonCameraDetectionModule::Initialize(const GeneralConfig& generalConfig, 
    const CrowdDetectionConfig& crowdConfig, 
    const PhotoLeakDetectionConfig& photoLeakConfig, 
    const AbsenceDetectionConfig& absenceConfig) {
    std::cout << "GeneralConfig details:" << std::endl;
    std::cout << "Camera Timeout: " << generalConfig.camera.timeout << std::endl;
    std::cout << "Camera Resolution: " << generalConfig.camera.width << "x" << generalConfig.camera.height << std::endl;
    std::cout << "Camera Frame Rate: " << generalConfig.camera.frame_rate << std::endl;
    std::cout << "Debug Mode: " << (generalConfig.debug_mode ? "Enabled" : "Disabled") << std::endl;
    m_general_config = generalConfig;
    m_crowd_config = crowdConfig;
    m_photo_leak_config = photoLeakConfig;
    m_absence_config = absenceConfig;

    // Initialize the camera
    DetectionError error = InitializeCamera(generalConfig);
    if (error != DetectionError::None) {
        std::cerr << "Failed to initialize camera." << std::endl
            << "Error code: " << static_cast<int>(error) << std::endl;
        return error;
    }
    // Initialize the detection graphs
    error = InitializeFaceDetectionGraph();
    if (error != DetectionError::None) {
        std::cerr << "Failed to initialize face detection graph." << std::endl;
        return error;
    }

    error = InitializeObjectDetectionGraph(photoLeakConfig);
    if (error != DetectionError::None) {
        std::cerr << "Failed to initialize object detection graph." << std::endl;
        return error;
    }
    std::cout << "Module initialized with camera timeout: " << generalConfig.camera.timeout
                << " and resolution:" << generalConfig.camera.width<<"x"<< 
                generalConfig.camera.height << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::StartDetection(const std::function<void(const DetectionResult&)>& callback) {
    if (m_detection_running) {
        std::cerr << "Detection is already running." << std::endl;
        return DetectionError::DetectionStartFailed;
    }
    m_detection_running = true;
 
    std::cout << "ResumeCameraCapture will started ." << std::endl;
    DetectionError err = ResumeCameraCapture();
    if (err != DetectionError::None) {
        std::cerr << "Failed to resume camera capture." << std::endl;
        return err;
    }
    std::cout << "AddOutputStreamPoller will started ." << std::endl;
    
    m_face_img_poller = 
        m_face_detection_graph->AddOutputStreamPoller(kOutputStream);
    if (!m_face_img_poller.ok())
    {
        std::cerr << "Failed to add output stream poller for face detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }

    m_face_detection_poller = 
        m_face_detection_graph->AddOutputStreamPoller(kFaceDetections);
    if (!m_face_detection_poller.ok())
    {
        std::cerr << "Failed to add output stream poller for face detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }
    std::cout << "m_face_poller AddOutputStreamPoller finished ." << std::endl;

    m_object_poller = 
        m_object_detection_graph->AddOutputStreamPoller(kOutputStream);
    if (!m_object_poller.ok())
    {
        std::cerr << "Failed to add output stream poller for object detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }

    m_object_direction_poller = m_object_detection_graph->AddOutputStreamPoller(kObjectDetections);
    if (!m_object_direction_poller.ok())
    {
        std::cerr << "Failed to add objection output stream direction poller for object detection." << std::endl;
        return DetectionError::MMPStartFailed;
    }

    std::cout << "m_object_poller AddOutputStreamPoller finished ." << std::endl;
    std::cout << "Detection will started ." << std::endl;
    auto status = m_face_detection_graph->StartRun({});
    if (!status.ok()) {
        std::cerr << "Failed to start face detection graph: " << status.message() << std::endl;
        return DetectionError::MMPStartFailed;
    }
    status = m_object_detection_graph->StartRun({});
    if (!status.ok()) {
        std::cerr << "Failed to start object detection graph: " << status.message() << std::endl;
        return DetectionError::MMPStartFailed;
    }
    // Start the detection thread
    startInterfaceDetectionThread(callback);
    std::cout << "Detection thread started successfully." << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::StopDetection() {
    if (!m_detection_running) {
        std::cerr << "Detection is not running." << std::endl;
        return DetectionError::None;
    }
    m_detection_running = false;
    stopInterfaceDetectionThread();
    std::cout << "Detection stopped." << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::DetectFromImage(const uint8_t* image_data, int width, int height, 
         const std::function<void(const DetectionResult&)>& callback) {
    if (!image_data || width <= 0 || height <= 0) {
        std::cerr << "Invalid image data." << std::endl;
        return DetectionError::ProcessingError;
    }

    // Simulate image detection
    DetectionResult result;
    result.person_count = 1;  // Example: detected one person
    callback(result);

    std::cout << "Image detection completed for resolution: " << width << "x" << height << std::endl;
    return DetectionError::None;
    }

    DetectionError PersonCameraDetectionModule::GetDetectionResult(DetectionResult& result) const {
    if (!m_detection_running) {
        std::cerr << "Detection is not running." << std::endl;
        return DetectionError::Uninitialized;
    }

    // Simulate returning a detection result
    result.person_count = 1;  // Example: detected one person
    result.is_absent = false;
    result.is_photo_leak_possible = false;

    std::cout << "Returning detection result." << std::endl;
    return DetectionError::None;
}

bool PersonCameraDetectionModule::IsDetectionRunning() const {
    return m_detection_running;
}


DetectionError PersonCameraDetectionModule::InitializeFaceDetectionGraph() {
    // Simulate initialization of the face detection graph
    std::cout << "Initializing face detection graph..." << std::endl;

    ABSL_LOG(INFO) << "Initialize the calculator graph.";
    std::string calculator_graph_config_contents = generateFaceDetectionGraph();
    mediapipe::CalculatorGraphConfig config =
                            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
                                                        calculator_graph_config_contents);

    m_face_detection_graph = std::make_unique<mediapipe::CalculatorGraph>();
    
    absl::Status status = m_face_detection_graph->Initialize(config);
    if (!status.ok()) {
        std::cerr << "Failed to initialize face detection graph: " << status.message() << std::endl;
        return DetectionError::MMPInitializationFailed;
    }

    std::cout << "Face detection graph initialized successfully." << std::endl;
    return DetectionError::None;
}

DetectionError PersonCameraDetectionModule::InitializeObjectDetectionGraph(const PhotoLeakDetectionConfig& photoLeakConfig) {
    // Simulate initialization of the object detection graph
    std::cout << "Initializing object detection graph..." << std::endl;

    std::string calculator_graph_config_contents = generateObjectionDetectionGraph(photoLeakConfig.pose_threshold,9);
    mediapipe::CalculatorGraphConfig config =
                            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
                                                        calculator_graph_config_contents);

    m_object_detection_graph = std::make_unique<mediapipe::CalculatorGraph>();
    
    absl::Status status = m_object_detection_graph->Initialize(config);
    if (!status.ok()) {
        std::cerr << "Failed to initialize face detection graph: " << status.message() << std::endl;
        return DetectionError::MMPInitializationFailed;
    }

    std::cout << "Object detection graph initialized successfully." << std::endl;
    return DetectionError::None;
}


DetectionError PersonCameraDetectionModule::InitializeCamera(const GeneralConfig& generalConfig) {
    std::cout << "Initializing camera with timeout: " << generalConfig.camera.timeout
              << " and resolution: " << generalConfig.camera.width<<"x"<<
              generalConfig.camera.height << std::endl;

    // Open the camera using OpenCV
    m_camera = std::make_unique<cv::VideoCapture>();  // Open default camera (index 0)
    std::cout << "Camera initialized successfully." << std::endl;
    return DetectionError::None;
}


DetectionError PersonCameraDetectionModule::PauseCameraCapture() {
    if (!m_camera || !m_camera->isOpened()) {
        std::cerr << "Camera is not initialized or already closed." << std::endl;
        return DetectionError::None;
    }

    if (m_camera->isOpened()) {
        m_camera->release(); // 释放摄像头资源
        return DetectionError::None;
    }
    std::cerr << "Camera capture is already paused." << std::endl;
    return DetectionError::InvalidOperation;
}


DetectionError PersonCameraDetectionModule::ResumeCameraCapture() {
    if (!m_camera) {
        std::cerr << "Failed to resume camera capture." << std::endl;
        return DetectionError::CameraInitializationFailed;
    }
    
    if (!m_camera->isOpened()) {
        std::cerr << "will open camera 0" << std::endl;

        bool camera_opened = false;
        for (int i = 0; i <= 10; ++i) {
          m_camera->open(i, cv::CAP_DSHOW);  // Use DirectShow backend explicitly for better compatibility.
          if (m_camera->isOpened()) {
            ABSL_LOG(INFO) << "Successfully opened webcam at index " << i;
            camera_opened = true;
            break;
          } else {
            ABSL_LOG(WARNING) << "Failed to open webcam at index " << i;
          }
        }
        if (!camera_opened) {
          ABSL_LOG(ERROR) << "Failed to open any webcam. Please check the device connections.";
          return DetectionError::CameraStartFailed;
        }

        std::cerr << "opened camera 0" << std::endl;
        if (!m_camera->isOpened()) {
            std::cerr << "Failed to resume camera capture." << std::endl;
            return DetectionError::CameraStartFailed;
        }
            // Set camera resolution
        m_camera->set(cv::CAP_PROP_FRAME_WIDTH, m_general_config.camera.width);
        m_camera->set(cv::CAP_PROP_FRAME_HEIGHT, m_general_config.camera.height);
        // Set camera frame rate
        m_camera->set(cv::CAP_PROP_FPS, m_general_config.camera.frame_rate);
        return DetectionError::None;
    }

    std::cerr << "Camera capture is already running." << std::endl;
    return DetectionError::None;
}

void PersonCameraDetectionModule::startInterfaceDetectionThread(const std::function<void(const DetectionResult&)>& callback) {
    if (m_inference_thread.joinable()) {
        std::cerr << "Detection thread is already running." << std::endl;
        return;
    }
    m_detection_running = true;
    if (m_general_config.debug_mode)
    {
        ABSL_LOG(INFO) << "Debug mode is enabled. Creating debug windows.";
        cv::namedWindow(kOriginWindow, /*flags=WINDOW_AUTOSIZE*/ 1);
        cv::namedWindow(kFaceDetection, /*flags=WINDOW_AUTOSIZE*/ 1);
        cv::namedWindow(kObjectDetection, /*flags=WINDOW_AUTOSIZE*/ 1);
    }
    
    m_inference_thread = std::thread([this, callback]() {
        while (m_detection_running) {
            if (!m_camera || !m_camera->isOpened()) {
                std::cerr << "Camera is not initialized or closed." << std::endl;
                break;
            }

            cv::Mat camera_frame_raw;
            *m_camera>>camera_frame_raw;
            if (camera_frame_raw.empty()) {
                ABSL_LOG(WARNING) << "Ignore empty frames from Queue.";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
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

            // Send image packet into the graph.
            size_t frame_timestamp_us =
                (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
            
            auto addRetStatus = m_face_detection_graph->AddPacketToInputStream(
                    kInputStream, mediapipe::Adopt(face_input_frame.release())
                                      .At(mediapipe::Timestamp(frame_timestamp_us)));
            auto addRetStatus2 = m_object_detection_graph->AddPacketToInputStream(
                    kInputStream, mediapipe::Adopt(object_input_frame.release())
                                      .At(mediapipe::Timestamp(frame_timestamp_us)));

            if (!addRetStatus.ok()||!addRetStatus2.ok()) {
                std::cerr << "Failed to send frame to face detection graph: "
                          << addRetStatus.message() << std::endl;
                continue;
            }
            // std::cout << "Processing frame at timestamp: " << frame_timestamp_us << " us" << std::endl;


            DetectionResult result;

            result.is_camera_blocked = IsCameraPossiblyBlocked(camera_frame);
            processFaceDetectionResult(result);
            processObjectDetectionResult(result);


            // Output detection result to the log
            std::cout << "Detection Result: " 
                      << "Person Count: " << result.person_count 
                      << ", Is Absent: " << result.is_absent 
                      << ", Absence Timeout: " << result.absence_timeout 
                      << ", Is Photo Leak Possible: " << result.is_photo_leak_possible 
                      << ", Is Camera Blocked: " << result.is_camera_blocked
                      << std::endl;

            // Invoke the callback with the detection result
            // callback(result);
            // Log detection results for debugging
            std::cout << "Processed frame at timestamp: " << frame_timestamp_us << " us" << std::endl;            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000/m_general_config.camera.frame_rate));
        }
    });

    std::cout << "Detection thread started." << std::endl;
}

void PersonCameraDetectionModule::stopInterfaceDetectionThread() {
    m_detection_running = false;
    if (m_inference_thread.joinable()) {
        m_inference_thread.join();
    }
    std::cout << "Detection thread stopped." << std::endl;
}


void PersonCameraDetectionModule::dealAbsentDetection(DetectionResult& result) {
    if (result.is_absent) {
        if (!m_already_recode_time) {
            m_already_recode_time = true;
            result.absence_timeout = 0;
            m_absence_start_time = std::chrono::steady_clock::now();
            std::cout << "Absence detected, starting timer." << std::endl;
        } else {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - m_absence_start_time).count();
            result.absence_timeout = elapsed_time;
            std::cout << "Absence detected, elapsed time: " << elapsed_time << " seconds." << std::endl;
        }
    } else {
        if (m_already_recode_time) {
            m_already_recode_time = false;
            std::cout << "Presence detected, resetting absence timer." << std::endl;
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
            }
        }
    } else {
        if(m_face_img_poller->QueueSize() > 0) {
            if (m_face_img_poller->Next(&face_packet)) {
                auto& output_frame = face_packet.Get<mediapipe::ImageFrame>();
            }
            
            result.person_count = 0;
            result.is_absent = true;
            dealAbsentDetection(result);
        }    
    }
}


void PersonCameraDetectionModule::processObjectDetectionResult(DetectionResult &result)
{
    // Poll object detection results
    // bool m_camera_detected = false;  // 是否检测到相机
    // std::chrono::steady_clock::time_point m_detection_start_time;  // 检测开始时间
    // std::chrono::steady_clock::time_point m_last_detection_time;  // 上次检测时间
    // int m_frame_counter = 0;  // 帧计数器
    if (m_object_direction_poller->QueueSize() > 0) {
        // std::cout << "m_object_direction_poller->QueueSize() > 0" << std::endl;
        mediapipe::Packet obj_packet;
        if (m_object_direction_poller->Next(&obj_packet)) {
            // Process face detection results
            for (const auto& detection : obj_packet.Get<std::vector<mediapipe::Detection>>()) 
            {
                const auto num_labels =
                    std::max(detection.label_size(), detection.label_id_size());
                if (num_labels > 0)
                {
                    if(!m_camera_detected ) {
                        m_camera_detected = true;
                        m_detection_start_time = std::chrono::steady_clock::now();
                        std::cout << "Camera detected, starting timer." << std::endl;
                        break;
                    }
                    else
                    {
                        auto current_time = std::chrono::steady_clock::now();
                        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - m_detection_start_time).count();
                        if (elapsed_time > CAMERA_DETECTION_TIMER) {
                            // Camera detected for too long, set to absent
                            result.is_photo_leak_possible = true;  // Example: set to true for now                       
                        }
                    }
                }
            }
        }
        
        mediapipe::Packet object_packet;
        if (m_object_poller->Next(&object_packet)) {
           auto& output_frame = object_packet.Get<mediapipe::ImageFrame>();
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
        }
    } 
}


bool PersonCameraDetectionModule::IsCameraPossiblyBlocked(cv::Mat image) const
{
    if (image.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return false;
    }

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray_image, mean, stddev);

    double variance = stddev[0] * stddev[0];
    if (variance < 200) {
        return true; // Camera is possibly blocked
    } else {
        return false; // Camera is not blocked
    }
}




extern "C" {

PERSON_CAMERA_DETECTION_MODULE_API IPersonCameraDetectionModule* CreatePersonCameraDetectionModule() {
    return new PersonCameraDetectionModule();
}

PERSON_CAMERA_DETECTION_MODULE_API void DestroyPersonCameraDetectionModule(IPersonCameraDetectionModule* module) {
    delete module;
}

};