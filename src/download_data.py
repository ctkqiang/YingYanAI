from icrawler.builtin import GoogleImageCrawler
import os
import sys
from typing import Optional


def validate_input(text: str) -> bool:
    """验证输入是否有效"""
    return bool(text and not text.isspace())


def download_images(
    keywords: str,
    max_num: int = 5,
    save_dir: str = "images",
    is_train: bool = False,
) -> Optional[str]:
    try:
        if is_train:
            save_dir = "images/train"

        # 创建主目录
        os.makedirs(save_dir, exist_ok=True)

        # 为关键词创建子目录
        keyword_dir = os.path.join(save_dir, keywords)
        os.makedirs(keyword_dir, exist_ok=True)

        # 自定义文件名解析器（使用索引）
        class CustomNameParser:
            def __init__(self, name):
                self.count = 1
                self.name = name

            def __call__(self, task, response):
                ext = os.path.splitext(task.file_name)[1] or ".jpg"
                filename = f"{self.name}_{self.count}{ext}"
                print(f"正在下载: {filename}")
                self.count += 1
                return filename

        # 配置爬虫
        crawler = GoogleImageCrawler(
            storage={"root_dir": keyword_dir},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )

        crawler.parser.filename_parser = CustomNameParser(name=keywords)

        print(f"\n正在为 '{keywords}' 下载 {max_num} 张图片...")
        crawler.crawl(keyword=keywords, max_num=max_num)

        return keyword_dir

    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


def main():
    # 从用户获取关键词
    keywords = input("请输入搜索关键词: ").strip()

    if not validate_input(keywords):
        print("错误: 请输入有效的关键词")
        sys.exit(1)

    # 获取需要下载的图片数量
    try:
        max_num = int(input("需要下载多少张图片？(默认: 5): ") or "5")
        if max_num <= 0:
            raise ValueError
    except ValueError:
        print("错误: 请输入有效的正整数")
        sys.exit(1)

    # 下载图片
    save_path = download_images(keywords, max_num)

    if save_path:
        print(f"\n下载完成！图片保存在: {save_path}")
    else:
        print("\n下载失败！")


if __name__ == "__main__":
    main()
