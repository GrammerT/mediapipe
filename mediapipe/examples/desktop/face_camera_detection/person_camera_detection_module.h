#pragma once

#include "mediapipe/framework/calculator_framework.h"
#include "include/IPersonCameraDetectionModule.h"
#include <memory>
#include <thread>
#include <chrono>

namespace mediapipe{
    class CalculatorGraph;
    class OutputStreamPoller;
} 
namespace cv{
    class VideoCapture;
}

class PersonCameraDetectionModule : public IPersonCameraDetectionModule {
public:
    PersonCameraDetectionModule() ;
    ~PersonCameraDetectionModule() ;

    DetectionError Initialize(const GeneralConfig& generalConfig, 
            const CrowdDetectionConfig& crowdConfig, 
            const PhotoLeakDetectionConfig& photoLeakConfig, 
            const AbsenceDetectionConfig& absenceConfig);


    DetectionError StartDetection(const std::function<void(const DetectionResult&)>& callback);

    DetectionError StopDetection();

    DetectionError DetectFromImage(const uint8_t* image_data, int width, int height, 
                        const std::function<void(const DetectionResult&)>& callback) ;

    DetectionError GetDetectionResult(DetectionResult& result) const;

    bool IsDetectionRunning() const ;

    DetectionError PauseCameraCapture() override;
    DetectionError ResumeCameraCapture() override;



private:
    DetectionError InitializeFaceDetectionGraph();
    DetectionError InitializeObjectDetectionGraph(const PhotoLeakDetectionConfig& photoLeakConfig);
    DetectionError InitializeCamera(const GeneralConfig& generalConfig);

    void startInterfaceDetectionThread(const std::function<void(const DetectionResult&)>& callback);
    void stopInterfaceDetectionThread();
    void dealAbsentDetection(DetectionResult& result);

    void processFaceDetectionResult(DetectionResult &result);
    void processObjectDetectionResult(DetectionResult &result);

private:
    bool m_detection_running = false;
    GeneralConfig m_general_config;
    CrowdDetectionConfig m_crowd_config;
    PhotoLeakDetectionConfig m_photo_leak_config;
    AbsenceDetectionConfig m_absence_config;

    std::unique_ptr<mediapipe::CalculatorGraph> m_face_detection_graph;
    std::unique_ptr<mediapipe::CalculatorGraph> m_object_detection_graph;

    absl::StatusOr<mediapipe::OutputStreamPoller> m_face_img_poller; 
    absl::StatusOr<mediapipe::OutputStreamPoller> m_face_detection_poller;
    
    absl::StatusOr<mediapipe::OutputStreamPoller> m_object_poller;
    absl::StatusOr<mediapipe::OutputStreamPoller> m_object_direction_poller;

    std::unique_ptr<cv::VideoCapture> m_camera;
    std::thread m_inference_thread;

    bool m_already_recode_time = false;
    std::chrono::time_point<std::chrono::steady_clock> m_absence_start_time;

    //! 记录检测到相机相关的逻辑
    bool m_camera_detected = false;  // 是否检测到相机
    std::chrono::steady_clock::time_point m_detection_start_time;  // 检测开始时间
};