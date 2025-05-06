#pragma once

#include <string>
#include <functional>

#ifdef _WIN32
    #ifdef PERSON_CAMERA_DETECTION_MODULE_EXPORT
        #define PERSON_CAMERA_DETECTION_MODULE_API __declspec(dllexport)
    #else
        #define PERSON_CAMERA_DETECTION_MODULE_API __declspec(dllimport)
    #endif
#else
    #define PERSON_CAMERA_DETECTION_MODULE_API
#endif


struct GeneralConfig {
    // 摄像头配置
    struct {
        int width = 640;         // 摄像头宽度
        int height = 480;        // 摄像头高度
        int frame_rate = 5;      // 摄像头帧率
        int timeout = 3000;      // 摄像头启动超时（毫秒）
    } camera;

    bool debug_mode = false;     // 是否开启调试模式-可以展示摄像头画面
    std::string log_path;        // 日志路径
};

struct DetectionConfig {
    bool isOpen;               // 是否开启此检测
};


//! 多人围观
struct CrowdDetectionConfig:DetectionConfig {
    int threshold = 2;            // 触发黑屏的人数阈值
    bool capture_photo = true;     // 是否拍摄现场照片
};

//! 拍照泄密
struct PhotoLeakDetectionConfig:DetectionConfig  {
    float pose_threshold = 0.2f;   // 拍照动作置信度阈值
};

//! 人员离席
struct AbsenceDetectionConfig :DetectionConfig {
    int timeout = 10;              // 离席判定时间（秒）
};


enum class DetectionError {
    None,                   // 无错误
    InitializationFailed,   // 初始化失败
    Uninitialized,          // 未初始化错误-请先初始化
    MMPInitializationFailed, // 推理引擎初始化失败
    MMPStartFailed,       // 推理引擎启动失败
    CameraInitializationFailed, // 摄像头初始化失败
    CameraStartFailed,      // 摄像头启动失败
    DetectionStartFailed,   // 启动检测失败
    ProcessingError,        // 图像处理错误
    InvalidOperation,      // 无效操作
    UnknownError            // 未知错误
};

struct DetectionResult {
    bool is_photo_leak_possible = false; // 是否可能在拍照
    bool is_camera_blocked = false;   // 摄像头是否可能被遮挡
    bool is_absent = false;        // 是否离席
    int  absence_timeout = 0;     // 离席超时时间（秒）
    int  person_count = 0;          // 检测到的人数
    
};


class PERSON_CAMERA_DETECTION_MODULE_API IPersonCameraDetectionModule {
public:
    virtual ~IPersonCameraDetectionModule() = default;

    virtual DetectionError Initialize(const GeneralConfig& generalConfig, 
                            const CrowdDetectionConfig& crowdConfig, 
                            const PhotoLeakDetectionConfig& photoLeakConfig, 
                            const AbsenceDetectionConfig& absenceConfig) = 0;

    
    virtual DetectionError StartDetection(const std::function<void(const DetectionResult&)>& callback) = 0;
    
    virtual DetectionError StopDetection() = 0;
    virtual bool IsDetectionRunning() const = 0;
    /**
     * @brief Perform detection on an externally provided image.
     * 
     * This method allows external images in RGB32 format to be passed for detection.
     * 
     * @param image_data Pointer to the image data in RGB32 format.
     * @param width Width of the image.
     * @param height Height of the image.
     * @return DetectionError Returns an error code indicating the result of the detection.
     */
    virtual DetectionError DetectFromImage(const uint8_t* image_data, int width, int height, 
                                           const std::function<void(const DetectionResult&)>& callback) = 0;

    /**
     * @brief Retrieve the latest detection result.
     * 
     * This method provides the most recent detection result after processing.
     * 
     * @param result Reference to a DetectionResult object to store the result.
     * @return DetectionError Returns an error code indicating the status of the retrieval.
     */
    virtual DetectionError GetDetectionResult(DetectionResult& result) const = 0;

    /**
     * @brief 
     * This method resumes the camera to start capturing frames after being paused.
     * 
     * @return DetectionError Returns an error code indicating the result of the operation.
     */
    virtual DetectionError ResumeCameraCapture() = 0;

    /**
     * @brief 
     * This method stops the camera from capturing frames temporarily.
     * 
     * @return DetectionError Returns an error code indicating the result of the operation.
     */
    virtual DetectionError PauseCameraCapture() = 0;

};

