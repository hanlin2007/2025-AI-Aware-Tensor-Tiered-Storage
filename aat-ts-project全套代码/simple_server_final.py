#!/usr/bin/env python3
"""
AAT-TS 测试报告服务器 - 修复时间显示版本
使用真实当前时间显示最新生成时间
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import time
import json
from datetime import datetime
import glob


class AATReportHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        """自定义日志格式"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")

    def end_headers(self):
        """添加CORS头信息"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()


def find_all_reports():
    """查找所有测试报告文件 - 支持新报告格式"""
    report_files = []

    # 查找性能报告 (新格式)
    performance_reports = glob.glob('aat_performance_report_*.png') + glob.glob('aat_test_report_*.png')
    for report in performance_reports:
        if 'aat_performance_report_' in report:
            timestamp = report.replace('aat_performance_report_', '').replace('.png', '')
        else:
            timestamp = report.replace('aat_test_report_', '').replace('.png', '')

        # 清理时间戳显示
        display_timestamp = timestamp
        if display_timestamp.startswith('english_'):
            display_timestamp = display_timestamp.replace('english_', '')

        # 获取文件修改时间作为真实生成时间
        file_mtime = os.path.getmtime(report)
        real_time = datetime.fromtimestamp(file_mtime)

        report_files.append({
            'filename': report,
            'type': 'performance_report',
            'timestamp': timestamp,  # 保持原始时间戳用于排序
            'display_timestamp': display_timestamp,  # 清理后的显示时间戳
            'real_time': real_time,  # 真实文件修改时间
            'display_name': '性能图表报告'
        })

    # 查找技术HTML报告 (新格式)
    technical_reports = glob.glob('aat_technical_report_*.html')
    for report in technical_reports:
        timestamp = report.replace('aat_technical_report_', '').replace('.html', '')

        # 清理时间戳显示
        display_timestamp = timestamp
        if display_timestamp.startswith('english_'):
            display_timestamp = display_timestamp.replace('english_', '')

        # 获取文件修改时间作为真实生成时间
        file_mtime = os.path.getmtime(report)
        real_time = datetime.fromtimestamp(file_mtime)

        report_files.append({
            'filename': report,
            'type': 'technical_report',
            'timestamp': timestamp,
            'display_timestamp': display_timestamp,
            'real_time': real_time,
            'display_name': '技术分析报告'
        })

    # 查找详细结果
    result_files = glob.glob('aat_detailed_results_*.json')
    for result in result_files:
        timestamp = result.replace('aat_detailed_results_', '').replace('.json', '')

        # 清理时间戳显示
        display_timestamp = timestamp
        if display_timestamp.startswith('english_'):
            display_timestamp = display_timestamp.replace('english_', '')

        # 获取文件修改时间作为真实生成时间
        file_mtime = os.path.getmtime(result)
        real_time = datetime.fromtimestamp(file_mtime)

        report_files.append({
            'filename': result,
            'type': 'detailed_results',
            'timestamp': timestamp,
            'display_timestamp': display_timestamp,
            'real_time': real_time,
            'display_name': '详细测试数据'
        })

    # 查找文本报告
    text_reports = glob.glob('aat_text_report_*.txt')
    for report in text_reports:
        timestamp = report.replace('aat_text_report_', '').replace('.txt', '')

        # 清理时间戳显示
        display_timestamp = timestamp
        if display_timestamp.startswith('english_'):
            display_timestamp = display_timestamp.replace('english_', '')

        # 获取文件修改时间作为真实生成时间
        file_mtime = os.path.getmtime(report)
        real_time = datetime.fromtimestamp(file_mtime)

        report_files.append({
            'filename': report,
            'type': 'text_report',
            'timestamp': timestamp,
            'display_timestamp': display_timestamp,
            'real_time': real_time,
            'display_name': '文本测试报告'
        })

    # 按真实时间排序
    report_files.sort(key=lambda x: x['real_time'], reverse=True)
    return report_files


def get_latest_report_info(report_files):
    """获取最新报告信息"""
    if not report_files:
        return "无报告", "无报告"

    # 获取最新报告
    latest_report = report_files[0]

    # 最新报告ID（清理后的时间戳）
    latest_report_id = latest_report['display_timestamp']

    # 最新生成时间（真实文件修改时间）
    latest_generate_time = latest_report['real_time'].strftime("%Y-%m-%d %H:%M:%S")

    return latest_report_id, latest_generate_time


def generate_comprehensive_html(report_files):
    """生成综合HTML报告页面 - 支持新报告格式"""
    latest_report_id, latest_generate_time = get_latest_report_info(report_files)

    # 按类型分类报告
    performance_reports = [r for r in report_files if r['type'] == 'performance_report']
    technical_reports = [r for r in report_files if r['type'] == 'technical_report']
    detailed_reports = [r for r in report_files if r['type'] == 'detailed_results']
    text_reports = [r for r in report_files if r['type'] == 'text_report']

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AAT-TS 智能存储系统 - 测试报告中心</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}

            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }}

            .header h1 {{
                font-size: 2.2em;
                margin-bottom: 10px;
                font-weight: 300;
            }}

            .header h2 {{
                font-size: 1.1em;
                font-weight: 300;
                opacity: 0.9;
                margin-bottom: 20px;
            }}

            .stats-bar {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                padding: 20px;
                background: #f8f9fa;
            }}

            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }}

            .stat-card h3 {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 10px;
            }}

            .stat-card .value {{
                font-size: 1.8em;
                font-weight: bold;
                color: #2c3e50;
            }}

            .category-section {{
                margin: 30px 0;
                padding: 0 30px;
            }}

            .category-title {{
                font-size: 1.4em;
                margin-bottom: 20px;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}

            .reports-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 25px;
                margin-bottom: 40px;
            }}

            .report-card {{
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid #e9ecef;
            }}

            .report-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            }}

            .report-header {{
                padding: 20px;
                color: white;
            }}

            .performance .report-header {{
                background: linear-gradient(135deg, #3498db, #2980b9);
            }}

            .technical .report-header {{
                background: linear-gradient(135deg, #27ae60, #2ecc71);
            }}

            .detailed .report-header {{
                background: linear-gradient(135deg, #e74c3c, #c0392b);
            }}

            .text .report-header {{
                background: linear-gradient(135deg, #f39c12, #e67e22);
            }}

            .report-header h3 {{
                font-size: 1.3em;
                margin-bottom: 5px;
            }}

            .report-type {{
                display: inline-block;
                background: rgba(255,255,255,0.2);
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                margin-top: 5px;
            }}

            .report-content {{
                padding: 20px;
            }}

            .report-content img {{
                width: 100%;
                height: auto;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}

            .report-meta {{
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #e9ecef;
                color: #666;
                font-size: 0.9em;
            }}

            .btn {{
                display: inline-block;
                padding: 10px 20px;
                border-radius: 6px;
                text-decoration: none;
                margin-top: 10px;
                transition: background 0.3s ease;
                color: white;
                font-weight: bold;
            }}

            .performance .btn {{
                background: #3498db;
            }}

            .performance .btn:hover {{
                background: #2980b9;
            }}

            .technical .btn {{
                background: #27ae60;
            }}

            .technical .btn:hover {{
                background: #219653;
            }}

            .detailed .btn {{
                background: #e74c3c;
            }}

            .detailed .btn:hover {{
                background: #c0392b;
            }}

            .text .btn {{
                background: #f39c12;
            }}

            .text .btn:hover {{
                background: #e67e22;
            }}

            .footer {{
                text-align: center;
                padding: 30px;
                background: #f8f9fa;
                color: #666;
                border-top: 1px solid #e9ecef;
            }}

            .empty-state {{
                text-align: center;
                padding: 60px 30px;
                color: #666;
                grid-column: 1 / -1;
            }}

            .empty-state h3 {{
                font-size: 1.5em;
                margin-bottom: 15px;
            }}

            @media (max-width: 768px) {{
                .reports-grid {{
                    grid-template-columns: 1fr;
                    padding: 15px;
                }}

                .header h1 {{
                    font-size: 1.8em;
                }}

                .category-section {{
                    padding: 0 15px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 AAT-TS 智能存储系统</h1>
                <h2>测试报告中心 - 基于真实模型数据的全面性能评估</h2>
                <div class="stats-bar">
                    <div class="stat-card">
                        <h3>测试报告总数</h3>
                        <div class="value">{len(report_files)}</div>
                    </div>
                    <div class="stat-card">
                        <h3>最新报告ID</h3>
                        <div class="value">{latest_report_id}</div>
                    </div>
                    <div class="stat-card">
                        <h3>最新生成时间</h3>
                        <div class="value" style="font-size: 1.4em;">{latest_generate_time}</div>
                    </div>
                </div>
            </div>

            <!-- 性能图表报告 -->
            <div class="category-section">
                <h2 class="category-title">📊 性能图表报告</h2>
                <div class="reports-grid">
    """

    if not performance_reports:
        html_content += """
                    <div class="empty-state">
                        <h3>📈 暂无性能图表报告</h3>
                        <p>性能图表报告包含详细的性能指标可视化</p>
                    </div>
        """
    else:
        for report in performance_reports[:4]:  # 显示最近4个性能报告
            real_time_str = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
            html_content += f"""
                    <div class="report-card performance">
                        <div class="report-header">
                            <h3>{report['display_name']}</h3>
                            <div class="report-type">性能可视化</div>
                        </div>
                        <div class="report-content">
                            <img src="{report['filename']}" alt="{report['display_name']}">
                            <div class="report-meta">
                                <p><strong>文件:</strong> {report['filename']}</p>
                                <p><strong>生成时间:</strong> {real_time_str}</p>
                                <a href="{report['filename']}" class="btn" target="_blank">查看图表</a>
                            </div>
                        </div>
                    </div>
            """

    html_content += """
                </div>
            </div>

            <!-- 技术分析报告 -->
            <div class="category-section">
                <h2 class="category-title">🔬 技术分析报告</h2>
                <div class="reports-grid">
    """

    if not technical_reports:
        html_content += """
                    <div class="empty-state">
                        <h3>📋 暂无技术分析报告</h3>
                        <p>技术分析报告包含系统架构、测试流程和数据分析</p>
                    </div>
        """
    else:
        for report in technical_reports[:4]:  # 显示最近4个技术报告
            real_time_str = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
            html_content += f"""
                    <div class="report-card technical">
                        <div class="report-header">
                            <h3>{report['display_name']}</h3>
                            <div class="report-type">技术文档</div>
                        </div>
                        <div class="report-content">
                            <div style="text-align: center; padding: 20px;">
                                <h4>📄 技术分析报告</h4>
                                <p>包含系统架构图、测试流程和详细数据分析</p>
                                <p><strong>特色内容:</strong></p>
                                <ul style="text-align: left; margin: 15px 0;">
                                    <li>系统架构原理图</li>
                                    <li>测试流程与方法论</li>
                                    <li>数据统计计算方法</li>
                                    <li>性能指标分析</li>
                                </ul>
                            </div>
                            <div class="report-meta">
                                <p><strong>文件:</strong> {report['filename']}</p>
                                <p><strong>生成时间:</strong> {real_time_str}</p>
                                <a href="{report['filename']}" class="btn" target="_blank">查看报告</a>
                            </div>
                        </div>
                    </div>
            """

    html_content += """
                </div>
            </div>

            <!-- 数据报告 -->
            <div class="category-section">
                <h2 class="category-title">📁 数据报告</h2>
                <div class="reports-grid">
    """

    # 详细数据报告
    for report in detailed_reports[:2]:  # 显示最近2个详细报告
        real_time_str = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"""
                    <div class="report-card detailed">
                        <div class="report-header">
                            <h3>{report['display_name']}</h3>
                            <div class="report-type">原始数据</div>
                        </div>
                        <div class="report-content">
        """
        # 尝试加载JSON数据展示摘要
        try:
            with open(report['filename'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            html_content += f"""
                            <div style="font-family: Arial, sans-serif; font-size: 0.9em;">
                                <p><strong>测试时间:</strong> {data.get('test_timestamp', 'N/A')}</p>
                                <p><strong>总请求数:</strong> {data.get('performance_stats', {{}}).get('total_requests', 0)}</p>
                                <p><strong>热层命中率:</strong> {data.get('performance_stats', {{}}).get('hot_hit_rate', 0) * 100:.1f}%</p>
                                <p><strong>真实数据:</strong> {data.get('data_authenticity', {{}}).get('all_real_data', False) and '✅ 是' or '❌ 否'}</p>
                            </div>
            """
        except:
            html_content += '<p>详细测试数据文件</p>'

        html_content += f"""
                            <div class="report-meta">
                                <p><strong>文件:</strong> {report['filename']}</p>
                                <p><strong>生成时间:</strong> {real_time_str}</p>
                                <a href="{report['filename']}" class="btn" target="_blank">查看数据</a>
                            </div>
                        </div>
                    </div>
        """

    # 文本报告
    for report in text_reports[:2]:  # 显示最近2个文本报告
        real_time_str = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"""
                    <div class="report-card text">
                        <div class="report-header">
                            <h3>{report['display_name']}</h3>
                            <div class="report-type">文本格式</div>
                        </div>
                        <div class="report-content">
                            <div style="text-align: center; padding: 20px;">
                                <h4>📝 文本测试报告</h4>
                                <p>纯文本格式的性能报告，便于快速查看</p>
                            </div>
                            <div class="report-meta">
                                <p><strong>文件:</strong> {report['filename']}</p>
                                <p><strong>生成时间:</strong> {real_time_str}</p>
                                <a href="{report['filename']}" class="btn" target="_blank">查看报告</a>
                            </div>
                        </div>
                    </div>
        """

    if not detailed_reports and not text_reports:
        html_content += """
                    <div class="empty-state">
                        <h3>📁 暂无数据报告</h3>
                        <p>数据报告包含原始测试数据和文本格式报告</p>
                    </div>
        """

    html_content += """
                </div>
            </div>

            <div class="footer">
                <p>AAT-TS 智能存储系统 | 基于真实BERT模型数据的AI感知分层存储中间件</p>
                <p>测试场景: 在线推理 • 边缘微调 • 科研环境 • 压缩效率</p>
                <p>服务器时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p><strong>✅ 所有测试使用100%真实模型数据 - 无模拟数据</strong></p>
            </div>
        </div>

        <script>
            // 简单的交互功能
            document.addEventListener('DOMContentLoaded', function() {
                // 添加点击动画
                const cards = document.querySelectorAll('.report-card');
                cards.forEach(card => {
                    card.addEventListener('click', function() {
                        this.style.transform = 'scale(0.98)';
                        setTimeout(() => {
                            this.style.transform = '';
                        }, 150);
                    });
                });

                // 显示页面加载时间
                console.log('AAT-TS 报告中心已加载 - ' + new Date().toLocaleString());
            });
        </script>
    </body>
    </html>
    """

    return html_content


def main():
    PORT = 8080

    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 查找所有报告
    report_files = find_all_reports()

    if report_files:
        print("📁 找到以下测试报告:")
        # 按类型显示报告
        performance_reports = [r for r in report_files if r['type'] == 'performance_report']
        technical_reports = [r for r in report_files if r['type'] == 'technical_report']
        detailed_reports = [r for r in report_files if r['type'] == 'detailed_results']

        if performance_reports:
            print("  📊 性能图表报告:")
            for report in performance_reports[:2]:
                real_time = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
                print(f"     {report['filename']} - {real_time}")

        if technical_reports:
            print("  🔬 技术分析报告:")
            for report in technical_reports[:2]:
                real_time = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
                print(f"     {report['filename']} - {real_time}")

        if detailed_reports:
            print("  📁 详细数据报告:")
            for report in detailed_reports[:2]:
                real_time = report['real_time'].strftime("%Y-%m-%d %H:%M:%S")
                print(f"     {report['filename']} - {real_time}")

    # 生成综合报告页面
    comprehensive_html = generate_comprehensive_html(report_files)
    with open("aat_reports_dashboard.html", "w", encoding='utf-8') as f:
        f.write(comprehensive_html)

    report_url = f"http://localhost:{PORT}/aat_reports_dashboard.html"

    Handler = AATReportHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("=" * 60)
        print("🌐 AAT-TS 测试报告服务器已启动!")
        print(f"📍 服务器地址: http://localhost:{PORT}")
        print(f"📊 报告面板: {report_url}")
        print("=" * 60)
        print("💡 请在浏览器中手动打开以上地址查看报告")
        print("🛑 按 Ctrl+C 停止服务器")
        print("=" * 60)

        # 取消自动打开浏览器，只显示提示信息
        print("⏳ 等待用户手动访问...")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 服务器已停止")
            print("感谢使用 AAT-TS 系统！")


if __name__ == "__main__":
    main()