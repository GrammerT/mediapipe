#pragma once
#include <memory>
#include <functional>
#include <string>

#define EFFECT_LIBRARY

#if defined(EFFECT_LIBRARY)
    #define EFFECT_API_EXPORT __declspec(dllexport)
#else
    #define EFFECT_API_EXPORT __declspec(dllimport)
#endif

#define MAX_DATA_NUM 8


/** return值说明
 * 0  : 成功
 * -1 : 参数异常
 * 
 */

enum class EPureColor
{
    kRED,
    kGREEN,
    kBLUE,
};
/**
 * @brief 视频特效参数
 * background_file_path只有在user_pure_color=false才生效
 * 
 */
struct SVideoEffectParam
{
    bool user_pure_color;
    EPureColor pure_color;
    std::string background_file_path ;
};

enum class EVideoFormat
{
    kRGB24,
    kYUV420P,
};
/**
 * @brief 推送给虚拟背景引擎的视频数据,建议rgba数据
 * 
 */
struct SVideoFrame
{
    EVideoFormat format;
    uint8_t* data[MAX_DATA_NUM];
    int linesize[MAX_DATA_NUM];
    int width;
    int height;
};


class EFFECT_API_EXPORT IVideoEffect
{
public:
    virtual ~IVideoEffect() {}

    /**
     * @brief 创建IVideoEffect实例
     * 
     * @return std::shared_ptr<IVideoEffect> 
     */
    static std::shared_ptr<IVideoEffect> create();

    /**
     * @brief 初始化 IVideoEffect, 失败后返回false
     * 
     * @param param 初始化参数
     * @return true 初始化成功
     * @return false 初始化失败
     */
    virtual bool initVideoEffect(std::shared_ptr<SVideoEffectParam> param) =0;

    // enable IVideoEffect, return 0 if failed
    /**
     * @brief 启用视频特效
     * 
     * @return 0-成功,其他失败
     */
    virtual int enableVideoEffect()=0;
    // disable IVideoEffect, return 0 if failed
    /**
     * @brief 关闭视频特效
     * 
     * @return 0-成功,其他失败
     */
    virtual int disableVideoEffect()=0;

    // 
    /**
     * @brief 推送frame到IVideoEffect,
     * 
     * @param frame 视频帧
     * @return 0-成功,其他失败 
     */
    virtual int pushVideoFrame(std::shared_ptr<SVideoFrame> frame) = 0;

    /**
     * @brief 设置视频帧接收回调
     * 
     * @param callback 回调
     */
    virtual void setVideoFrameReceiverCallback(std::function<void(std::shared_ptr<SVideoFrame>)> callback) = 0;

};