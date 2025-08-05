# 嘌呤助手微信小程序

## 项目说明

这是一个基于web-view的微信小程序，通过嵌入H5页面来提供完整的嘌呤分析功能。

## 项目结构

```
miniprogram/
├── app.js                 # 小程序入口文件
├── app.json              # 小程序配置文件
├── sitemap.json          # 站点地图配置
├── project.config.json   # 项目配置
├── pages/
│   ├── index/            # 首页
│   │   ├── index.wxml    # 页面结构
│   │   ├── index.js      # 页面逻辑
│   │   └── index.wxss    # 页面样式
│   └── webview/          # web-view页面
│       ├── webview.wxml  # 页面结构
│       ├── webview.js    # 页面逻辑
│       └── webview.wxss  # 页面样式
└── images/               # 图片资源
    └── logo.png          # 应用logo
```

## 使用步骤

### 1. 获取服务器公网IP

在服务器上运行：
```bash
curl ifconfig.me
```

### 2. 修改配置

1. 在 `app.js` 中修改 `serverUrl` 为你的服务器公网IP：
```javascript
globalData: {
  serverUrl: 'http://你的服务器公网IP:3000'
}
```

2. 在 `project.config.json` 中填入你的小程序AppID：
```json
{
  "appid": "你的小程序AppID"
}
```

### 3. 配置服务器

1. 确保前端服务监听0.0.0.0：
```bash
cd project
npm run build
npm start
```

2. 确保服务器防火墙开放3000端口

### 4. 配置微信小程序后台

1. 登录微信公众平台
2. 进入"开发管理" -> "开发设置"
3. 在"业务域名"中添加你的H5域名（如：http://你的服务器IP:3000）

### 5. 上传小程序

1. 用微信开发者工具打开miniprogram文件夹
2. 填入AppID
3. 点击"上传"按钮

## 注意事项

1. 确保H5页面能在手机浏览器中正常访问
2. web-view只能访问已配置的业务域名
3. 建议使用HTTPS协议以提高安全性
4. 如果使用域名，需要配置SSL证书

## 功能特性

- 美观的启动页面
- 无缝嵌入H5应用
- 支持分享功能
- 响应式设计 