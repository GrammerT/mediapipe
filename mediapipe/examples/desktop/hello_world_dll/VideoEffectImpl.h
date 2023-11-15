#pragma once

#include "IVideoEffect.h"

class VideoEffectImpl:public IVideoEffect
{

public:
    VideoEffectImpl();
    ~VideoEffectImpl();
    virtual bool initVideoEffect(std::shared_ptr<SVideoEffectParam> param) override;
    virtual int enableVideoEffect()override;
    virtual int disableVideoEffect()override;
    virtual int pushVideoFrame(std::shared_ptr<SVideoFrame> frame) override;
    virtual void setVideoFrameReceiverCallback(std::function<void(std::shared_ptr<SVideoFrame>)> callback) override;

private:
    bool m_is_enable=false;
    std::shared_ptr<SVideoEffectParam> m_param=nullptr;
    std::function<void(std::shared_ptr<SVideoFrame>)> m_receiver_callback=nullptr;
};

