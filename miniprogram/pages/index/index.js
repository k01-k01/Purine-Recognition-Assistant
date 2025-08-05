const app = getApp()

Page({
  data: {
    // 页面数据
  },

  onLoad() {
    // 页面加载时执行
  },

  goToWebView() {
    // 跳转到web-view页面
    wx.navigateTo({
      url: '/pages/webview/webview'
    })
  }
}) 