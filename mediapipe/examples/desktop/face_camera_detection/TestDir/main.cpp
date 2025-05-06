#include <windows.h>
#include <iostream>
#include <functional>
#include "..//include//IPersonCameraDetectionModule.h"


typedef IPersonCameraDetectionModule* (*CreateModuleFunc)();
typedef void (*DestroyModuleFunc)(IPersonCameraDetectionModule*);

int main(int argc, char** argv) {
  
    //加载 DLL
    if (GetFileAttributesA("C:/Users/qq675/_bazel_qq675/zskgtuew/execroot/mediapipe/bazel-out/x64_windows-opt/bin/mediapipe/examples/desktop/face_camera_detection/person_camera_detection_module.dll") == INVALID_FILE_ATTRIBUTES) {
        std::cerr << "DLL file not found!" << std::endl;
        return -1;
    }
    SetCurrentDirectoryA("C:/Users/qq675/_bazel_qq675/zskgtuew/execroot/mediapipe/bazel-out/x64_windows-opt/bin/mediapipe/examples/desktop/face_camera_detection");
    HMODULE hModule = LoadLibraryA("C:/Users/qq675/_bazel_qq675/zskgtuew/execroot/mediapipe/bazel-out/x64_windows-opt/bin/mediapipe/examples/desktop/face_camera_detection/person_camera_detection_module.dll");  
    if (!hModule) {
        DWORD errorCode = GetLastError();
        LPVOID errorMsg;
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            errorCode,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPSTR)&errorMsg,
            0,
            NULL);
        std::cerr << "Failed to load DLL! Error code: " << errorCode << ", Message: " << (char*)errorMsg << std::endl;
        LocalFree(errorMsg);
        return -1;
    }

    // 获取导出函数地址
    CreateModuleFunc CreateModule = (CreateModuleFunc)GetProcAddress(hModule, "CreatePersonCameraDetectionModule");
    DestroyModuleFunc DestroyModule = (DestroyModuleFunc)GetProcAddress(hModule, "DestroyPersonCameraDetectionModule");

    if (!CreateModule || !DestroyModule) {
        std::cerr << "Failed to get function addresses!" << std::endl;
        FreeLibrary(hModule);
        return -1;
    }

    // 创建模块实例
    IPersonCameraDetectionModule* module = CreateModule();
    if (!module) {
        std::cerr << "Failed to create module instance!" << std::endl;
        FreeLibrary(hModule);
        return -1;
    }

    // 配置模块
    GeneralConfig generalConfig;
    generalConfig.camera.timeout = 3000;
	generalConfig.camera.width = 640;
	generalConfig.camera.height = 480;
    generalConfig.debug_mode = false;


    CrowdDetectionConfig crowdConfig;
    crowdConfig.isOpen = true;
    crowdConfig.threshold = 3;
    crowdConfig.capture_photo = true;

    PhotoLeakDetectionConfig photoLeakConfig;
    photoLeakConfig.isOpen = true;
    photoLeakConfig.pose_threshold = 0.35f;

    AbsenceDetectionConfig absenceConfig;
    absenceConfig.isOpen = true;
    absenceConfig.timeout = 15;

    if (module->Initialize(generalConfig, crowdConfig, photoLeakConfig, absenceConfig) != DetectionError::None) {
        std::cerr << "Failed to initialize module!" << std::endl;
        DestroyModule(module);
        FreeLibrary(hModule);
        return -1;
    }

    // 设置检测回调
    auto detectionCallback = [](const DetectionResult& result) {
        std::cout << "Detection Result: " << std::endl;
        std::cout << "  Person Count: " << result.person_count << std::endl;
        std::cout << "  Is Absent: " << (result.is_absent ? "Yes" : "No") << std::endl;
        std::cout << "  Photo Leak Possible: " << (result.is_photo_leak_possible ? "Yes" : "No") << std::endl;
    };

    // 启动检测
    if (module->StartDetection(detectionCallback) != DetectionError::None) {
        std::cerr << "Failed to start detection!" << std::endl;
        DestroyModule(module);
        FreeLibrary(hModule);
        return -1;
    }

    // 模拟运行一段时间
    Sleep(90000);

    // 停止检测
    if (module->StopDetection() != DetectionError::None) {
        std::cerr << "Failed to stop detection!" << std::endl;
    }

    // 获取最新检测结果
    DetectionResult result;
    if (module->GetDetectionResult(result) == DetectionError::None) {
        std::cout << "Latest Detection Result: " << std::endl;
        std::cout << "  Person Count: " << result.person_count << std::endl;
        std::cout << "  Is Absent: " << (result.is_absent ? "Yes" : "No") << std::endl;
        std::cout << "  Photo Leak Possible: " << (result.is_photo_leak_possible ? "Yes" : "No") << std::endl;
    } else {
        std::cerr << "Failed to retrieve detection result!" << std::endl;
    }

    // 销毁模块实例
    DestroyModule(module);

    // 卸载 DLL
    FreeLibrary(hModule);

    return EXIT_SUCCESS;
}
