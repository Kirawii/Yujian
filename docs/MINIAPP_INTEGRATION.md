# 小程序端接入指南

## 概述

后端服务支持两种模式，通过 `REVIEW_MODE` 环境变量控制：

| 模式 | 后端配置 | 小程序表现 |
|------|----------|-----------|
| **审核模式** | `REVIEW_MODE=true` | 显示图片序列（规避视频审核限制） |
| **正常模式** | `REVIEW_MODE=false` | 播放视频 |

---

## 1. API 接口变更

### 新接口（推荐）

```
POST /translate/video_with_media
```

**功能**：上传视频进行手语翻译，根据后端模式返回图片或视频

**请求参数**：
- `file`: 视频文件 (multipart/form-data)

**返回值**（审核模式 - JSON）：
```json
{
  "success": true,
  "text": "吃药了吗?现在身体怎么样?",
  "message": "翻译成功",
  "media_type": "images",
  "review_mode": true,
  "images": [
    "/9j/4AAQSkZJRgABAQ...",  // base64 图片1
    "/9j/4AAQSkZJRgABAQ...",  // base64 图片2
    "/9j/4AAQSkZJRgABAQ...",  // base64 图片3
    "/9j/4AAQSkZJRgABAQ...",  // base64 图片4
    "/9j/4AAQSkZJRgABAQ..."   // base64 图片5
  ]
}
```

**返回值**（正常模式 - 文件流）：
- 直接返回视频文件 (Content-Type: video/mp4)

---

## 2. 小程序端代码修改

### 2.1 修改翻译上传逻辑

```javascript
// utils/api.js

const API_BASE = 'https://your-domain.com';  // 替换为实际域名

/**
 * 上传视频进行手语翻译
 * @param {string} videoPath - 本地视频路径
 * @returns {Promise<{text: string, mediaType: 'images'|'video', data: any}>}
 */
async function translateVideo(videoPath) {
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: `${API_BASE}/translate/video_with_media`,
      filePath: videoPath,
      name: 'file',
      success: (res) => {
        // 判断返回类型
        const contentType = res.header['Content-Type'] || res.header['content-type'];
        
        if (contentType && contentType.includes('application/json')) {
          // 审核模式 - JSON 返回
          const data = JSON.parse(res.data);
          resolve({
            text: data.text,
            mediaType: 'images',
            data: data.images  // base64 图片数组
          });
        } else {
          // 正常模式 - 视频文件
          // 保存临时文件路径
          const fs = wx.getFileSystemManager();
          const tempPath = `${wx.env.USER_DATA_PATH}/temp_video_${Date.now()}.mp4`;
          
          fs.writeFile({
            filePath: tempPath,
            data: res.data,
            encoding: 'binary',
            success: () => {
              resolve({
                text: '',  // 需要从其他接口获取，或后端调整返回头
                mediaType: 'video',
                data: tempPath
              });
            },
            fail: reject
          });
        }
      },
      fail: reject
    });
  });
}

module.exports = { translateVideo };
```

### 2.2 修改展示组件

```vue
<!-- pages/translate/result.vue -->
<template>
  <view class="result-page">
    <!-- 翻译文本 -->
    <view class="text-result">{{ translatedText }}</view>
    
    <!-- 审核模式：图片轮播 -->
    <view v-if="mediaType === 'images'" class="image-container">
      <swiper class="swiper" indicator-dots autoplay circular>
        <swiper-item v-for="(img, index) in imageList" :key="index">
          <image :src="img" mode="aspectFit" class="slide-image"/>
          <text class="image-index">{{ index + 1 }} / {{ imageList.length }}</text>
        </swiper-item>
      </swiper>
      <text class="hint">审核模式：显示关键帧</text>
    </view>
    
    <!-- 正常模式：视频播放 -->
    <view v-else-if="mediaType === 'video'" class="video-container">
      <video
        :src="videoUrl"
        controls
        class="video-player"
        object-fit="contain"
      />
    </view>
    
    <!-- 其他功能按钮 -->
    <button @click="playTTS">朗读翻译结果</button>
    <button @click="saveResult">保存结果</button>
  </view>
</template>

<script>
import { translateVideo } from '@/utils/api.js';

export default {
  data() {
    return {
      translatedText: '',
      mediaType: '',  // 'images' 或 'video'
      imageList: [],  // base64 图片数组
      videoUrl: ''    // 视频临时路径
    };
  },
  
  async onLoad(options) {
    const videoPath = options.videoPath;
    
    try {
      uni.showLoading({ title: '翻译中...' });
      
      const result = await translateVideo(videoPath);
      
      this.translatedText = result.text;
      this.mediaType = result.mediaType;
      
      if (result.mediaType === 'images') {
        // 审核模式：将 base64 转为本地临时路径
        this.imageList = await this.convertBase64ToTempPaths(result.data);
      } else {
        // 正常模式：直接使用视频路径
        this.videoUrl = result.data;
      }
      
    } catch (error) {
      uni.showToast({ title: '翻译失败', icon: 'none' });
      console.error(error);
    } finally {
      uni.hideLoading();
    }
  },
  
  methods: {
    /**
     * 将 base64 图片数组转为本地临时路径
     */
    async convertBase64ToTempPaths(base64Array) {
      const fs = wx.getFileSystemManager();
      const tempPaths = [];
      
      for (let i = 0; i < base64Array.length; i++) {
        const base64 = base64Array[i];
        const tempPath = `${wx.env.USER_DATA_PATH}/frame_${i}_${Date.now()}.jpg`;
        
        // 注意：微信小程序 base64 需要去除前缀
        const pureBase64 = base64.replace(/^data:image\/\w+;base64,/, '');
        
        try {
          fs.writeFileSync(tempPath, pureBase64, 'base64');
          tempPaths.push(tempPath);
        } catch (e) {
          console.error('保存图片失败:', e);
        }
      }
      
      return tempPaths;
    },
    
    /**
     * 播放 TTS 语音
     */
    playTTS() {
      // 调用 TTS 接口
      wx.request({
        url: `${API_BASE}/tts`,
        method: 'POST',
        data: {
          text: this.translatedText,
          voice: '2222'
        },
        success: (res) => {
          // 播放返回的音频...
        }
      });
    }
  }
};
</script>

<style scoped>
.text-result {
  font-size: 24px;
  font-weight: bold;
  padding: 20px;
  text-align: center;
}

.swiper {
  height: 400px;
}

.slide-image {
  width: 100%;
  height: 100%;
}

.image-index {
  position: absolute;
  bottom: 10px;
  right: 10px;
  background: rgba(0,0,0,0.5);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.hint {
  text-align: center;
  color: #999;
  font-size: 12px;
  padding: 10px;
}

.video-player {
  width: 100%;
  height: 400px;
}
</style>
```

---

## 3. 注意事项

### 3.1 文件大小限制

| 限制项 | 数值 | 说明 |
|--------|------|------|
| 上传视频 | 50MB | 超过需压缩 |
| 返回图片 | 5张 | 均匀采样的关键帧 |
| base64 编码 | 增大 33% | 图片比原文件大 |

### 3.2 审核期提示文案建议

在审核模式的图片展示区域，建议添加提示：

```html
<view class="review-hint">
  <text>当前为演示模式，展示手语关键帧</text>
  <text>实际使用中可播放完整视频</text>
</view>
```

### 3.3 调试技巧

**查看当前模式**：
```javascript
wx.request({
  url: 'https://your-domain.com/review_mode',
  success: (res) => {
    console.log('当前模式:', res.data.review_mode ? '审核' : '正常');
  }
});
```

---

## 4. 审核提交清单

提交微信小程序审核前，请确认：

- [ ] 后端 `REVIEW_MODE` 设置为 `true`
- [ ] 小程序端能正确显示图片轮播
- [ ] 不展示任何视频播放组件
- [ ] 翻译功能正常可用
- [ ] 添加适当的提示文案

审核通过后：

- [ ] 后端 `REVIEW_MODE` 设置为 `false`
- [ ] 重启后端服务
- [ ] 验证视频播放功能
- [ ] 发布新版本小程序（如有需要）

---

## 5. 问题排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| 图片不显示 | base64 格式错误 | 检查是否去除 `data:image/jpeg;base64,` 前缀 |
| 视频无法播放 | 临时文件路径错误 | 确保使用 `wx.env.USER_DATA_PATH` 下的路径 |
| 返回格式不对 | 后端模式未切换 | 检查 `/review_mode` 返回值，重启服务 |
| 翻译失败 | 姿态检测不到 | 确保视频中有清晰的手语动作 |

---

## 6. 联系人

后端开发：[待填写]

有任何问题请及时沟通！
