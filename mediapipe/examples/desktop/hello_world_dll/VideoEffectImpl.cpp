#include "VideoEffectImpl.h"


std::shared_ptr<IVideoEffect> IVideoEffect::create()
{   
    return std::make_shared<VideoEffectImpl>();
}


VideoEffectImpl::VideoEffectImpl()
{
}

VideoEffectImpl::~VideoEffectImpl()
{
}

bool VideoEffectImpl::initVideoEffect(std::shared_ptr<SVideoEffectParam> param)
{
    if (!param)
    {
        return false;
    }
    m_param = param;
    return true;
}

int VideoEffectImpl::enableVideoEffect()
{
    m_is_enable = true;
    
    return 0;
}

int VideoEffectImpl::disableVideoEffect()
{
    m_is_enable = false;

    return 0;
}

int VideoEffectImpl::pushVideoFrame(std::shared_ptr<SVideoFrame> frame)
{
    if (m_param)
    {
        return -1;
    }
    
    return 0;
}

void VideoEffectImpl::setVideoFrameReceiverCallback(std::function<void(std::shared_ptr<SVideoFrame>)> callback)
{
    m_receiver_callback = callback;
}
