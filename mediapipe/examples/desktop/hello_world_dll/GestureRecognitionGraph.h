#ifndef GESTURE_RECOGNITION_GRAPH_H
#define GESTURE_RECOGNITION_GRAPH_H

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/calculators/image/VirtualBackground_calculator.pb.h"
#include "mediapipe/calculators/util/annotation_overlay_calculator.pb.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include <atomic>
#include <thread>

class GestureRecognitionGraph {
public:
    GestureRecognitionGraph();
    ~GestureRecognitionGraph();

    absl::Status Initialize();
    absl::Status Run();
    absl::Status Close();

    void pushVideoFrame(cv::Mat frame);
    bool getThumbUp() { 
        auto ret = m_thumbup;
        this->m_thumbup = false;
        return ret; }
private:
    bool dealGestureResult(mediapipe::Packet &packet_landmarks,cv::Mat camera_frame);
    

private:
    mediapipe::CalculatorGraph m_recognition_graph;
    std::unique_ptr<mediapipe::OutputStreamPoller> m_pPoller;
    std::unique_ptr<mediapipe::OutputStreamPoller> m_pPoller_landmarks;
    std::atomic_bool m_is_graph_running=false;
    std::thread m_graph_thread;

    std::mutex m_mat_mutex;
    cv::Mat m_current_mat;
    bool m_thumbup = false;
};

#endif // GESTURE_RECOGNITION_GRAPH_H