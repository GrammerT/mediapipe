#include "GestureRecognitionGraph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "InternalDefine.h"
#include "hand_gesture_recognition.h"
#include "hand_tracking_data.h"

std::string calculator_graph_config_contents_gesture = str_gesture_recognition;

constexpr char kInputStream1[] =  "input_video";
GestureRecognitionGraph::GestureRecognitionGraph() {}

GestureRecognitionGraph::~GestureRecognitionGraph() {
    Close();
}

absl::Status GestureRecognitionGraph::Initialize() {
    ABSL_LOG(INFO) << "will init gesture graph config.";
    mediapipe::CalculatorGraphConfig config =mediapipe::ParseTextProtoOrDie
                                                    <mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents_gesture);
    
    
    // 初始化图
    auto status = m_recognition_graph.Initialize(config);
    if(!status.ok())
    {
      ABSL_LOG(ERROR) << "media pipe recognition graph init error : "<<status.ToString();
      return  absl::OkStatus();
    }
	// 添加video输出流
	auto sop = m_recognition_graph.AddOutputStreamPoller("output_video");
    if (!sop.ok()) {
        // 处理错误，例如打印错误消息
        ABSL_LOG(ERROR) << "Error: " << sop.status().ToString();
        // 返回特定错误码或状态
        return absl::OkStatus();
    }

	m_pPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));

    // 	// 添加landmarks输出流
    mediapipe::StatusOrPoller sop_landmark = m_recognition_graph.AddOutputStreamPoller("landmarks");
    if (!sop_landmark.ok()) {
        // 处理错误，例如打印错误消息
        ABSL_LOG(ERROR) << "Error: " << sop_landmark.status().ToString();
        // 返回特定错误码或状态
        return absl::OkStatus();
    }
    m_pPoller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_landmark.value()));
    // m_pPoller_landmarks->SetMaxQueueSize(1);
    ABSL_LOG(INFO) << "finished init gesture graph config.";

    return absl::OkStatus();
}

absl::Status GestureRecognitionGraph::Run() {
    if (m_is_graph_running) {
        return absl::OkStatus();
    }
    ABSL_LOG(INFO) << "Starting graph.";
    m_is_graph_running = true;

    // 启动处理线程
    m_graph_thread = std::thread([this]() {

        auto status = m_recognition_graph.StartRun({});
        if(!status.ok())
        {
            ABSL_LOG(INFO) << "Starting graph false: " << m_graph_thread.get_id();
            return ;
        }
        ABSL_LOG(INFO) << "Starting graph success: " << m_graph_thread.get_id();

        while (m_is_graph_running) {
            std::unique_lock<std::mutex> locker(m_mat_mutex);
            cv::Mat camera_frame = m_current_mat;
            locker.unlock();
            if (camera_frame.empty())
            {
                ABSL_LOG(WARNING) << "Ignore empty frames from Queue.";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // Wrap Mat into an ImageFrame.
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary);
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            camera_frame.copyTo(input_frame_mat);
            // ABSL_LOG(INFO) <<"camera_frame copy finished.";
            // Send image packet into the graph.

            size_t frame_timestamp_us =
                (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
            auto addRetStatus = m_recognition_graph.AddPacketToInputStream(
                kInputStream1, mediapipe::Adopt(input_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us)));
            // ABSL_LOG(INFO) <<"AddPacketToInputStream finished.";
            if (!addRetStatus.ok())
            {
                ABSL_LOG(WARNING) <<"addRetStatus return false."<<addRetStatus.ToString();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            // ABSL_LOG(INFO) <<"will call  m_pPoller_landmarks next.";
            mediapipe::Packet packet;
            if (!m_pPoller->Next(&packet)) 
            {
                // ABSL_LOG(WARNING) <<"m_pPoller->Next return false.";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (m_pPoller_landmarks->QueueSize() > 0) 
            {
                mediapipe::Packet packet_landmarks;
                if(m_pPoller_landmarks->Next(&packet_landmarks)) {
                    // ABSL_LOG(INFO) << "m_pPoller_landmarks->Next(&packet_landmarks)";
                    m_thumbup = dealGestureResult(packet_landmarks,camera_frame);
                    // ABSL_LOG(INFO) <<  "dealGestureResult return thumbup:"<<m_thumbup;
                }
                // ABSL_LOG(INFO) <<"finished call  m_pPoller_landmarks next.";
            }
            else
            {

            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    });

    return absl::OkStatus();
}

absl::Status GestureRecognitionGraph::Close() {
    if (!m_is_graph_running) {
        return absl::OkStatus();
    }

    m_is_graph_running = false;

    if (m_graph_thread.joinable()) {
        m_graph_thread.join();
    }

    MP_RETURN_IF_ERROR(m_recognition_graph.CloseInputStream("hand_landmarks"));
    MP_RETURN_IF_ERROR(m_recognition_graph.WaitUntilDone());

    return absl::OkStatus();
}

void GestureRecognitionGraph::pushVideoFrame(cv::Mat frame)
{
    std::lock_guard<std::mutex> locker(m_mat_mutex);
    m_current_mat = frame;
}

bool GestureRecognitionGraph::dealGestureResult(mediapipe::Packet &packet_landmarks,cv::Mat camera_frame)
{
  std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

  std::vector<PoseInfo> hand_landmarks;
  hand_landmarks.clear();
  bool haveThumbUp = false;
  for (int m = 0; m < output_landmarks.size(); ++m)
  {
    mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList = output_landmarks[m];

    std::vector<PoseInfo> singleHandGestureInfo;
    singleHandGestureInfo.clear();

    for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i)
    {
      PoseInfo info;
      const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
      info.x = landmark.x() * camera_frame.cols;
      info.y = landmark.y() * camera_frame.rows;
      singleHandGestureInfo.push_back(info);
      hand_landmarks.push_back(info);
    }

    HandGestureRecognition handGestureRecognition;
    int result = handGestureRecognition.GestureRecognition(singleHandGestureInfo);
    if(result == Gesture::ThumbUp)
    {
      haveThumbUp = true;
    //   ABSL_LOG(INFO) << "gesture recognition result is : "<<result;
    }
  }
  return haveThumbUp;
}
