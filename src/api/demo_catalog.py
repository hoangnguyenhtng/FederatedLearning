"""
Curated e-commerce metadata for demo: titles, descriptions, and category-matched images.
Embeddings still come from federated client data; display fields are enriched here.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

# Unsplash CDN — stable URLs, w=400 for product cards
CATEGORY_PRODUCTS: Dict[str, List[Dict[str, Any]]] = {
    "All_Beauty": [
        {
            "title": "Son Dưỡng Môi Hồng Tự Nhiên",
            "description": "Son dưỡng chiết xuất thiên nhiên, giúp môi mềm mịn và hồng hào. Không paraben.",
            "image_url": "https://images.unsplash.com/photo-1631214524020-8e787a24f826?w=400&h=400&fit=crop",
            "price_vnd": 185000,
        },
        {
            "title": "Serum Vitamin C 20% Dưỡng Sáng",
            "description": "Serum Vitamin C làm sáng da, mờ thâm và kích thích collagen.",
            "image_url": "https://images.unsplash.com/photo-1620916566398-39f1144ab7be?w=400&h=400&fit=crop",
            "price_vnd": 450000,
        },
        {
            "title": "Kem Chống Nắng SPF 50+ PA++++",
            "description": "Chống nắng phổ rộng UVA/UVB, kết cấu nhẹ không bết dính.",
            "image_url": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?w=400&h=400&fit=crop",
            "price_vnd": 320000,
        },
        {
            "title": "Mặt Nạ Collagen Dưỡng Ẩm",
            "description": "Mặt nạ collagen giúp da căng bóng sau một lần sử dụng.",
            "image_url": "https://images.unsplash.com/photo-1570194062560-ede2898be425?w=400&h=400&fit=crop",
            "price_vnd": 89000,
        },
        {
            "title": "Nước Tẩy Trang Micellar 500ml",
            "description": "Làm sạch makeup nhẹ nhàng, phù hợp da nhạy cảm.",
            "image_url": "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
            "price_vnd": 215000,
        },
        {
            "title": "Phấn Phủ Kiềm Dầu 24h",
            "description": "Finish matte tự nhiên, kiềm dầu suốt ngày.",
            "image_url": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
            "price_vnd": 275000,
        },
        {
            "title": "Bộ Cọ Trang Điểm 12 Cây",
            "description": "Lông cọ mềm, đủ bước makeup chuyên nghiệp.",
            "image_url": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=400&h=400&fit=crop",
            "price_vnd": 650000,
        },
        {
            "title": "Dầu Gội Dược Liệu Ngăn Rụng Tóc",
            "description": "Thảo dược tự nhiên, giảm rụng và kích thích mọc tóc.",
            "image_url": "https://images.unsplash.com/photo-1535585209827-a68fc5e0d31d?w=400&h=400&fit=crop",
            "price_vnd": 195000,
        },
        {
            "title": "Cushion Foundation SPF 50+",
            "description": "Che phủ tốt, finish bán lì, bảo vệ da cả ngày.",
            "image_url": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400&h=400&fit=crop",
            "price_vnd": 520000,
        },
        {
            "title": "Xịt Khoáng Cấp Ẩm Tức Thì",
            "description": "Cấp ẩm và làm dịu da mọi lúc mọi nơi.",
            "image_url": "https://images.unsplash.com/photo-1608248543809-9d8874a66e0e?w=400&h=400&fit=crop",
            "price_vnd": 145000,
        },
        {
            "title": "Kem Dưỡng Ban Đêm Retinol",
            "description": "Giảm nếp nhăn, phục hồi da khi ngủ.",
            "image_url": "https://images.unsplash.com/photo-1570194062560-ede2898be425?w=400&h=400&fit=crop",
            "price_vnd": 380000,
        },
        {
            "title": "Toner Hoa Hồng Cân Bằng pH",
            "description": "Se khít lỗ chân lông, cân bằng độ ẩm sau rửa mặt.",
            "image_url": "https://images.unsplash.com/photo-1608571423902-eed4a5ad8108?w=400&h=400&fit=crop",
            "price_vnd": 165000,
        },
    ],
    "Video_Games": [
        {
            "title": "PlayStation 5 DualSense Controller",
            "description": "Haptic feedback và adaptive triggers cho trải nghiệm chơi chân thực.",
            "image_url": "https://images.unsplash.com/photo-1606144042614-b2417e99c432?w=400&h=400&fit=crop",
            "price_vnd": 1850000,
        },
        {
            "title": "Nintendo Switch OLED",
            "description": "Màn hình OLED 7 inch, chơi mọi lúc mọi nơi.",
            "image_url": "https://images.unsplash.com/photo-1578303512597-81e6ccedad56?w=400&h=400&fit=crop",
            "price_vnd": 8500000,
        },
        {
            "title": "Tai Nghe Gaming RGB 7.1",
            "description": "Âm thanh vòm 7.1, micro khử ồn, đèn RGB.",
            "image_url": "https://images.unsplash.com/photo-1546435770-5efc71a8619e?w=400&h=400&fit=crop",
            "price_vnd": 950000,
        },
        {
            "title": "Bàn Phím Cơ Cherry MX RGB",
            "description": "Switch Cherry MX, khung nhôm, full RGB.",
            "image_url": "https://images.unsplash.com/photo-1511467687858-23d96bd43ef6?w=400&h=400&fit=crop",
            "price_vnd": 2200000,
        },
        {
            "title": "Chuột Gaming Wireless 25600 DPI",
            "description": "Siêu nhẹ 63g, pin 70 giờ, sensor cao cấp.",
            "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&h=400&fit=crop",
            "price_vnd": 1450000,
        },
        {
            "title": "Steam Deck 512GB",
            "description": "Máy chơi cầm tay chạy thư viện Steam đầy đủ.",
            "image_url": "https://images.unsplash.com/photo-1621259182978-fbf931544153?w=400&h=400&fit=crop",
            "price_vnd": 14500000,
        },
        {
            "title": "Ghế Gaming Ergonomic Pro",
            "description": "Tựa lưng ergonomic, đệm memory foam.",
            "image_url": "https://images.unsplash.com/photo-1598550476439-6847785fcea6?w=400&h=400&fit=crop",
            "price_vnd": 4200000,
        },
        {
            "title": "Tay Cầm Xbox Wireless",
            "description": "Grip chống trơn, tương thích PC và Xbox.",
            "image_url": "https://images.unsplash.com/photo-1592842130787-526511a47e22?w=400&h=400&fit=crop",
            "price_vnd": 1650000,
        },
        {
            "title": "Màn Hình Gaming 27\" 165Hz",
            "description": "QHD IPS, 1ms, G-Sync compatible.",
            "image_url": "https://images.unsplash.com/photo-1527443224754-9a86bfcbb13e?w=400&h=400&fit=crop",
            "price_vnd": 6800000,
        },
        {
            "title": "Bàn Di Chuột XXL RGB",
            "description": "900x400mm, viền LED, bề mặt micro-texture.",
            "image_url": "https://images.unsplash.com/photo-1618381131359-3819cfe56069?w=400&h=400&fit=crop",
            "price_vnd": 380000,
        },
        {
            "title": "Webcam 4K60 Streaming",
            "description": "HDR, tương thích OBS và phần mềm stream.",
            "image_url": "https://images.unsplash.com/photo-1587825140708-dfafb9cc4cdc?w=400&h=400&fit=crop",
            "price_vnd": 2100000,
        },
        {
            "title": "Đĩa Game PS5 God of War",
            "description": "Phiên bản đĩa Blu-ray cho PlayStation 5.",
            "image_url": "https://images.unsplash.com/photo-1493711662062-fa541f87f42e?w=400&h=400&fit=crop",
            "price_vnd": 1250000,
        },
    ],
    "Amazon_Fashion": [
        {
            "title": "Áo Thun Oversize Cotton Basic",
            "description": "Cotton 100%, form rộng unisex, nhiều màu.",
            "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=400&fit=crop",
            "price_vnd": 250000,
        },
        {
            "title": "Quần Jogger Streetwear",
            "description": "Vải tech-fleece co giãn, ống bo thời trang.",
            "image_url": "https://images.unsplash.com/photo-1473966968600-fa801b279a01?w=400&h=400&fit=crop",
            "price_vnd": 450000,
        },
        {
            "title": "Giày Sneaker Trắng Classic",
            "description": "Đế cao su bền, phối đồ dễ dàng.",
            "image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400&h=400&fit=crop",
            "price_vnd": 1250000,
        },
        {
            "title": "Kính Mát Polarized UV400",
            "description": "Khung nhẹ, chống tia UV, phong cách hiện đại.",
            "image_url": "https://images.unsplash.com/photo-1572635196233-8f7293b228f1?w=400&h=400&fit=crop",
            "price_vnd": 380000,
        },
        {
            "title": "Túi Tote Canvas Minimalist",
            "description": "Canvas dày, phù hợp đi học và đi làm.",
            "image_url": "https://images.unsplash.com/photo-1594223274512-ad4803739db8?w=400&h=400&fit=crop",
            "price_vnd": 195000,
        },
        {
            "title": "Áo Khoác Bomber Unisex",
            "description": "Lót lụa mềm, thêu logo tinh tế.",
            "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=400&fit=crop",
            "price_vnd": 680000,
        },
        {
            "title": "Đồng Hồ Minimalist Rose Gold",
            "description": "Mặt 38mm, dây da Italy, máy Nhật.",
            "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=400&fit=crop",
            "price_vnd": 1850000,
        },
        {
            "title": "Nón Bucket Hat Cotton",
            "description": "Phong cách Y2K, chống nắng nhẹ.",
            "image_url": "https://images.unsplash.com/photo-1588850561407-ed78c282e820?w=400&h=400&fit=crop",
            "price_vnd": 165000,
        },
        {
            "title": "Balo Laptop Chống Nước 15.6\"",
            "description": "Oxford chống nước, ngăn laptop, cổng USB.",
            "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=400&fit=crop",
            "price_vnd": 520000,
        },
        {
            "title": "Váy Midi Floral Mùa Hè",
            "description": "Chiffon bay bổng, họa tiết hoa vintage.",
            "image_url": "https://images.unsplash.com/photo-1595777453555-468d03cb3b7d?w=400&h=400&fit=crop",
            "price_vnd": 420000,
        },
        {
            "title": "Quần Jeans Slim Fit Xanh",
            "description": "Denim co giãn nhẹ, form ôm gọn.",
            "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&h=400&fit=crop",
            "price_vnd": 590000,
        },
        {
            "title": "Áo Sơ Mi Oxford Trắng",
            "description": "Vải Oxford cao cấp, đi làm và dạo phố.",
            "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b56?w=400&h=400&fit=crop",
            "price_vnd": 350000,
        },
    ],
    "Baby_Products": [
        {
            "title": "Bình Sữa Anti-Colic 240ml",
            "description": "PPSU an toàn, núm ty silicone mềm tự nhiên.",
            "image_url": "https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2?w=400&h=400&fit=crop",
            "price_vnd": 285000,
        },
        {
            "title": "Tã Dán Sơ Sinh Size S",
            "description": "Siêu thấm 12h, mỏng nhẹ, không hăm tã.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 320000,
        },
        {
            "title": "Xe Đẩy Gấp Gọn 5.5kg",
            "description": "Gấp một tay, giảm xóc, mái che UV50+.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 3500000,
        },
        {
            "title": "Đồ Chơi Xếp Hình Gỗ Montessori",
            "description": "Phát triển tư duy cho bé 1–3 tuổi.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 195000,
        },
        {
            "title": "Sữa Tắm Gội 2in1 Cho Bé",
            "description": "Công thức không nước mắt, pH 5.5.",
            "image_url": "https://images.unsplash.com/photo-1584515935687-775824f4571b?w=400&h=400&fit=crop",
            "price_vnd": 145000,
        },
        {
            "title": "Ghế Ăn Dặm 3 in 1",
            "description": "Ghế cao, thấp và booster — khay tháo rời.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 1850000,
        },
        {
            "title": "Máy Hút Sữa Điện Đôi",
            "description": "9 mức hút, massage kích sữa, motor êm.",
            "image_url": "https://images.unsplash.com/photo-1584515935687-775824f4571b?w=400&h=400&fit=crop",
            "price_vnd": 2450000,
        },
        {
            "title": "Yếm Ăn Silicon Chống Thấm",
            "description": "Food-grade, có máng hứng, dễ vệ sinh.",
            "image_url": "https://images.unsplash.com/photo-1584515935687-775824f4571b?w=400&h=400&fit=crop",
            "price_vnd": 89000,
        },
        {
            "title": "Đèn Ngủ Chiếu Sao Cho Bé",
            "description": "6 chế độ màu, nhạc ru, hẹn giờ tắt.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 250000,
        },
        {
            "title": "Bộ Chén Bát Ăn Dặm Lúa Mì",
            "description": "5 món từ sợi lúa mì, an toàn cho bé.",
            "image_url": "https://images.unsplash.com/photo-1584515935687-775824f4571b?w=400&h=400&fit=crop",
            "price_vnd": 125000,
        },
        {
            "title": "Nôi Điện Tự Đưa Thông Minh",
            "description": "Phát hiện khóc, 5 tốc độ, nhạc ru tích hợp.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 1950000,
        },
        {
            "title": "Ghế Ô Tô ISOFIX An Toàn",
            "description": "Tiêu chuẩn EU, gắn nhanh ISOFIX.",
            "image_url": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400&h=400&fit=crop",
            "price_vnd": 4200000,
        },
    ],
}

CLIENT_CATEGORY_MAP = {
    **{i: "All_Beauty" for i in range(0, 10)},
    **{i: "Video_Games" for i in range(10, 20)},
    **{i: "Amazon_Fashion" for i in range(20, 30)},
    **{i: "Baby_Products" for i in range(30, 40)},
}

CLIENT_DISPLAY = {
    "All_Beauty": ("FedShop Beauty", "💄", "Làm đẹp"),
    "Video_Games": ("FedShop GameZone", "🎮", "Video Games"),
    "Amazon_Fashion": ("FedShop Fashion", "👗", "Thời trang"),
    "Baby_Products": ("FedShop Baby", "🍼", "Mẹ & bé"),
}


def client_category(client_id: int) -> str:
    return CLIENT_CATEGORY_MAP.get(client_id, "All_Beauty")


def client_store_label(client_id: int) -> str:
    cat = client_category(client_id)
    brand, icon, vi = CLIENT_DISPLAY.get(cat, ("FedShop", "🛍️", cat))
    branch = (client_id % 10) + 1
    return f"{icon} {brand} — Chi nhánh {branch}"


def _stable_index(item_id: str, n: int) -> int:
    if n <= 0:
        return 0
    h = int(hashlib.md5(str(item_id).encode()).hexdigest(), 16)
    return h % n


# Màu + nhãn cho placeholder SVG (fallback cuối, không phụ thuộc CDN ngoài)
_CATEGORY_PLACEHOLDER = {
    "All_Beauty": ("fd79a8", "Làm đẹp"),
    "Video_Games": ("6c5ce7", "Games"),
    "Amazon_Fashion": ("00cec9", "Fashion"),
    "Baby_Products": ("fdcb6e", "Mẹ & Bé"),
}


def _picsum_url(category: str, item_id: str) -> str:
    """Ảnh deterministic theo item — picsum ổn định, ít bị 404 như Unsplash."""
    seed = int(hashlib.md5(f"{category}:{item_id}".encode()).hexdigest()[:12], 16)
    return f"https://picsum.photos/seed/fedshop_{category}_{seed}/400/400"


def _placeholder_svg_url(category: str) -> str:
    """Data-URI SVG — luôn hiển thị được khi mọi CDN ngoài lỗi."""
    color, label = _CATEGORY_PLACEHOLDER.get(category, ("6c5ce7", "Sản phẩm"))
    import base64

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">'
        f'<rect width="400" height="400" fill="#{color}"/>'
        f'<text x="50%" y="48%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="#ffffff" font-family="Arial,sans-serif" font-size="22" font-weight="600">'
        f"{label}</text>"
        f'<text x="50%" y="58%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="#ffffff" font-size="14" opacity="0.85">FedShop</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def resolve_product_image_url(
    category: str,
    item_id: str,
    *,
    preferred_url: Optional[str] = None,
    slot_index: Optional[int] = None,
) -> str:
    """
    Thứ tự ưu tiên: URL Amazon thật (nếu có) → picsum theo item_id → SVG nội bộ.
    Không dùng Unsplash làm URL chính (hay 403/404 trên một số mạng).
    """
    if preferred_url:
        u = str(preferred_url).strip()
        if u.startswith("http://") or u.startswith("https://"):
            return u

    return _picsum_url(category, item_id)


def enrich_product_metadata(
    item_id: str,
    category: str,
    *,
    fallback_title: Optional[str] = None,
    fallback_price: Optional[int] = None,
    fallback_rating: int = 4,
    fallback_image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Map federated item_id to curated display fields (same category)."""
    pool = CATEGORY_PRODUCTS.get(category) or CATEGORY_PRODUCTS["All_Beauty"]
    idx = _stable_index(item_id, len(pool))
    entry = pool[idx]
    rating = min(max(fallback_rating, 1), 5)
    return {
        "id": item_id,
        "title": fallback_title or entry["title"],
        "description": entry["description"],
        "category": category,
        "category_vi": CLIENT_DISPLAY.get(category, ("", "", category))[2],
        "price": fallback_price if fallback_price else entry["price_vnd"],
        "rating": float(rating),
        "review_count": 50 + (_stable_index(item_id, 450)),
        "image_url": resolve_product_image_url(
            category,
            item_id,
            preferred_url=fallback_image_url,
            slot_index=idx,
        ),
        "image_fallback": _placeholder_svg_url(category),
    }


def list_demo_clients() -> List[Dict[str, Any]]:
    out = []
    for cid in range(40):
        cat = client_category(cid)
        brand, icon, vi = CLIENT_DISPLAY[cat]
        out.append(
            {
                "client_id": cid,
                "category": cat,
                "category_vi": vi,
                "store_name": client_store_label(cid),
                "brand": brand,
                "icon": icon,
                "branch": (cid % 10) + 1,
            }
        )
    return out
