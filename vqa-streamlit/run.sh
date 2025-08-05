#!/bin/bash
# 检查是否已安装依赖
if ! pip show streamlit &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# 启动Streamlit应用
streamlit run app.py --server.port 8501 --server.address localhost
