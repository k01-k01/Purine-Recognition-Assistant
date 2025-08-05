App({
  onLaunch() {
    // 小程序启动时执行
    console.log('小程序启动')
  },
  globalData: {
    // 你的服务器公网IP地址，需要替换为实际的IP
    serverUrl: 'https://08891e98c139.ngrok-free.app'
  }
}) 