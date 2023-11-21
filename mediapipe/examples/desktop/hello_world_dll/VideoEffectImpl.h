#pragma once

#include <queue>
#include <mutex>
#include <memory>
#include <string>

#include <thread>
#include <condition_variable>
#include "IVideoEffect.h"
#include "mediapipe/framework/calculator_framework.h"
#include <opencv2/opencv.hpp>

#define SHOW_CV_WINDOW

class MemoryPool;

class VideoEffectImpl:public IVideoEffect
{

public:
    VideoEffectImpl();
    ~VideoEffectImpl();
    virtual bool initVideoEffect(std::shared_ptr<SVideoEffectParam> param) override;
    virtual void enableLogOutput(bool enable,std::string log_file_name)override;
    virtual int enableVideoEffect()override;
    virtual int disableVideoEffect()override;
    virtual int pushVideoFrame(std::shared_ptr<SVideoFrame> frame) override;
    virtual void setVideoFrameReceiverCallback(std::function<void(std::shared_ptr<SVideoFrame>)> callback) override;


private:
    void startGraphThread();
    void stopGraphThread();

    cv::Mat& PopVideoFrameQueueToCVMat();
    std::shared_ptr<SVideoFrame> matToSVideoFrame(const cv::Mat& inputMat, EVideoFormat format);
private:
    bool m_is_enable=false;
    std::shared_ptr<SVideoEffectParam> m_param=nullptr;
    std::function<void(std::shared_ptr<SVideoFrame>)> m_receiver_callback=nullptr;
    mediapipe::CalculatorGraph m_media_pipe_graph;
    absl::StatusOr<mediapipe::OutputStreamPoller> m_stream_poller;


    std::atomic_bool m_is_graph_running=false;
    std::thread m_graph_thread;

    std::queue<std::shared_ptr<SVideoFrame>> m_frame_queue;
    std::mutex m_frame_queue_mutex;
    std::condition_variable m_frame_queue_cond;

    std::shared_ptr<MemoryPool> m_memory_pool=nullptr;
    std::shared_ptr<SVideoFrame> m_yuv_2_rgb_tmpframe=nullptr; //! 临时做缓存用,不考虑此变量格式
#ifdef SHOW_CV_WINDOW
    cv::VideoCapture m_capture;
#endif
};

