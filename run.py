import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich import box
import sys
import os

console = Console()


def display_banner():
    banner = """
    鹰眼AI系统控制台
    =================
    """
    console.print(Panel(banner, style="bold green"))


def display_menu():
    table = Table(box=box.ROUNDED)
    table.add_column("选项", style="cyan")
    table.add_column("描述", style="yellow")

    table.add_row("1", "启动API服务")
    table.add_row("2", "启动实时监控")
    table.add_row("3", "训练模型")
    table.add_row("4", "系统状态")
    table.add_row("q", "退出系统")

    console.print(table)


def start_api():
    console.print("[bold green]正在启动API服务...[/]")
    current_dir = os.getcwd()
    try:
        os.chdir("src")
        sys.path.append(os.getcwd())  # Add src directory to Python path
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    finally:
        sys.path.remove(os.getcwd())  # Clean up Python path
        os.chdir(current_dir)  # Restore original directory


def start_monitor():
    # 打印启动信息
    console.print("[bold green]正在启动实时监控...[/]")
    os.system("python3 tools/live.py")


def train_model():
    console.print("[bold green]正在启动模型训练...[/]")
    os.system("python3 src/YingYanAI.py")


def show_status():
    status = Table(title="系统状态")
    status.add_column("组件", style="cyan")
    status.add_column("状态", style="green")

    model_exists = os.path.exists("models/yingyan_model.h5")
    status.add_row("AI模型", "已加载" if model_exists else "未加载")
    status.add_row("API服务", "就绪")
    status.add_row("实时监控", "就绪")

    console.print(status)


def main():
    try:
        while True:
            console.clear()
            display_banner()
            display_menu()

            choice = Prompt.ask("请选择操作", choices=["1", "2", "3", "4", "q"])

            if choice == "1":
                start_api()
            elif choice == "2":
                start_monitor()
            elif choice == "3":
                train_model()
            elif choice == "4":
                show_status()
                input("\n按回车键继续...")
            elif choice.lower() == "q":
                console.print("[bold red]正在退出系统...[/]")
                break

    except KeyboardInterrupt:
        console.print("\n[bold red]用户中断，正在退出...[/]")
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/]")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
