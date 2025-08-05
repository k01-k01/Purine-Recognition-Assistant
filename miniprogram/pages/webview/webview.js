const app = getApp()

Page({
  data: {
    webViewUrl: ''
  },

  onLoad() {
    // 设置web-view的URL
    this.setData({
      webViewUrl: app.globalData.serverUrl
    })
  },

  onShareAppMessage() {
    return {
      title: '嘌呤助手 - 智能食物识别',
      path: '/pages/index/index'
    }
  }
}) 