// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <windows.h>
#include <iostream>
#include "face_camera_detection/include/IPersonCameraDetectionModule.h"


typedef IPersonCameraDetectionModule* (*CreateModuleFunc)();
typedef void (*DestroyModuleFunc)(IPersonCameraDetectionModule*);

int main(int argc, char** argv) {
  
    // 加载 DLL
    HMODULE hModule = LoadLibraryA("person_camera_detection_module.dll");
    if (!hModule) {
        std::cerr << "Failed to load DLL!" << std::endl;
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

    // 调用接口类的函数
    GeneralConfig config;
    config.camera.timeout = 3000;
    config.camera.resolution = "1920x1080";
    module->Initialize(config);

    module->StartDetection();
    if (module->IsDetectionRunning()) {
        std::cout << "Detection is running!" << std::endl;
    }
    module->StopDetection();

    // 销毁模块实例
    DestroyModule(module);

    // 卸载 DLL
    FreeLibrary(hModule);

    return EXIT_SUCCESS;
}
