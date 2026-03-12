from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
import os

def fetch_images(dish_name, bing_count=200, google_count=100):
    dish_folder = dish_name.replace(" ", "_")  # safe folder name
    root_dir = os.path.join("dataset_week5", dish_folder)
    os.makedirs(root_dir, exist_ok=True)

    # Bing images
    bing_crawler = BingImageCrawler(storage={'root_dir': os.path.join(root_dir, "bing")})
    bing_crawler.crawl(keyword=dish_name, max_num=bing_count)

    # Google images
    google_crawler = GoogleImageCrawler(storage={'root_dir': os.path.join(root_dir, "google")})
    google_crawler.crawl(keyword=dish_name, max_num=google_count)

    print(f"✅ {dish_name} → {bing_count + google_count} images saved.")

def main():
    with open(r"E:\Smart_Nutrition_Detection\name.txt", "r", encoding="utf-8") as f:

        dish_names = [line.strip() for line in f if line.strip()]

    for dish in dish_names:
        fetch_images(dish, bing_count=200, google_count=100)

if __name__ == "__main__":
    main()