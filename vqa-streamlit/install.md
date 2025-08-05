# Streamlit BLIP视觉问答系统配置

## requirements.txt
```txt
streamlit==1.28.0
transformers==4.35.0
torch==2.0.1
torchvision==0.15.2
pillow==10.0.0
```

## 快速启动脚本 (run.sh / run.bat)

### Linux/Mac (run.sh)
```bash
#!/bin/bash
# 检查是否已安装依赖
if ! pip show streamlit &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# 启动Streamlit应用
streamlit run app.py --server.port 8501 --server.address localhost
```

### Windows (run.bat)
```batch
@echo off
REM 检查是否已安装依赖
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM 启动Streamlit应用
streamlit run app.py --server.port 8501 --server.address localhost
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行应用
```bash
streamlit run app.py
```

或者直接运行启动脚本：
- Linux/Mac: `./run.sh`
- Windows: `run.bat`

### 3. 访问应用
浏览器会自动打开，如果没有，请访问：http://localhost:8501

## 项目结构
```
blip-vqa-streamlit/
├── app.py              # 主应用文件（上面的代码）
├── requirements.txt    # Python依赖
├── run.sh             # Linux/Mac启动脚本
└── run.bat            # Windows启动脚本
```

## 功能特点

1. **一键部署**：只需要一个Python文件即可运行
2. **自动缓存**：模型只需加载一次，使用`@st.cache_resource`装饰器
3. **友好界面**：
   - 拖拽上传图片
   - 预设问题快速选择
   - 实时显示处理状态
   - 侧边栏显示帮助信息

4. **错误处理**：完善的错误提示和处理机制
5. **响应式设计**：适配不同屏幕尺寸

## 对比Flask方案的优势

| 特性 | Streamlit | Flask |
|------|-----------|--------|
| 代码量 | 1个文件，约200行 | 至少5个文件，500+行 |
| 部署难度 | 极简单 | 需要配置前后端 |
| 开发时间 | 30分钟 | 2-3小时 |
| 实时更新 | 自动热重载 | 需要手动刷新 |
| UI组件 | 内置丰富组件 | 需要自己写HTML/CSS |
| 文件上传 | 一行代码 | 需要处理表单和AJAX |

## 进阶优化建议

### 1. 添加更多模型选项
```python
model_name = st.selectbox(
    "选择模型",
    ["Salesforce/blip-vqa-base", 
     "Salesforce/blip-vqa-capfilt-large",
     "Salesforce/blip-image-captioning-base"]
)
```

### 2. 支持批量处理
```python
uploaded_files = st.file_uploader(
    "选择图片", 
    type=['jpg', 'png'],
    accept_multiple_files=True
)
```

### 3. 添加历史记录
```python
if 'history' not in st.session_state:
    st.session_state.history = []

# 保存问答记录
st.session_state.history.append({
    'image': image,
    'question': question,
    'answer': answer,
    'timestamp': datetime.now()
})
```

### 4. 导出功能
```python
if st.button("导出结果"):
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False)
    st.download_button(
        label="下载CSV",
        data=csv,
        file_name='vqa_results.csv',
        mime='text/csv'
    )
```

### 5. 部署到云端
- **Streamlit Cloud**: 最简单，直接从GitHub部署
- **Hugging Face Spaces**: 免费GPU支持
- **Docker部署**:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```