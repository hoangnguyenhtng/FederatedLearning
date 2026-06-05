/* ========================================
   FedShop - E-Commerce Application
   Complete JS with API + Mock Data Fallback
   ======================================== */

// ─── CONSTANTS ───────────────────────────────────────
function resolveApiBase() {
  const { protocol, hostname, port, origin } = window.location;
  if (protocol === 'file:' || !hostname) {
    return 'http://localhost:8000';
  }
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    if (port === '8000') return origin;
    return `${protocol}//${hostname}:8000`;
  }
  return origin;
}

const API_BASE = resolveApiBase();
const SESSION_STORAGE_KEY = 'fedshop_auth_session';

const STATIC_DEMO_CLIENTS = (() => {
  const brands = [
    ['All_Beauty', 'FedShop Beauty', '💄', 'Làm đẹp'],
    ['Video_Games', 'FedShop GameZone', '🎮', 'Video Games'],
    ['Amazon_Fashion', 'FedShop Fashion', '👗', 'Thời trang'],
    ['Baby_Products', 'FedShop Baby', '🍼', 'Mẹ & bé']
  ];
  const list = [];
  for (let cid = 0; cid < 40; cid++) {
    const [cat, brand, icon, vi] = brands[Math.floor(cid / 10)];
    const branch = (cid % 10) + 1;
    list.push({
      client_id: cid,
      category: cat,
      category_vi: vi,
      store_name: `${icon} ${brand} — Chi nhánh ${branch}`,
      brand,
      icon,
      branch
    });
  }
  return list;
})();

const CATEGORIES = [
  { id: 'All_Beauty',      name: 'Làm Đẹp',    icon: '💄', color: '#fd79a8' },
  { id: 'Video_Games',     name: 'Video Games',  icon: '🎮', color: '#6c5ce7' },
  { id: 'Amazon_Fashion',  name: 'Thời Trang',   icon: '👗', color: '#00cec9' },
  { id: 'Baby_Products',   name: 'Mẹ & Bé',     icon: '🍼', color: '#fdcb6e' }
];

// ─── MOCK DATA (40+ Products) ────────────────────────
const MOCK_PRODUCTS = [
  // ── All_Beauty (10) ──
  { id: 'b001', title: 'Son Dưỡng Môi Hồng Tự Nhiên', category: 'All_Beauty', price: 185000, rating: 4.5, review_count: 234, description: 'Son dưỡng môi chiết xuất thiên nhiên, giúp đôi môi mềm mịn và hồng hào tự nhiên. Không chứa paraben, an toàn cho làn da nhạy cảm.', image_url: 'https://picsum.photos/seed/beauty1/400/400' },
  { id: 'b002', title: 'Serum Vitamin C Dưỡng Sáng Da', category: 'All_Beauty', price: 450000, rating: 4.8, review_count: 567, description: 'Serum Vitamin C 20% giúp làm sáng da, mờ thâm nám và tăng cường collagen. Phù hợp mọi loại da.', image_url: 'https://picsum.photos/seed/beauty2/400/400' },
  { id: 'b003', title: 'Kem Chống Nắng SPF 50+ PA++++', category: 'All_Beauty', price: 320000, rating: 4.7, review_count: 891, description: 'Kem chống nắng phổ rộng bảo vệ da khỏi tia UVA/UVB. Kết cấu nhẹ, không bết dính, phù hợp dùng hàng ngày.', image_url: 'https://picsum.photos/seed/beauty3/400/400' },
  { id: 'b004', title: 'Mặt Nạ Collagen Vàng 24K', category: 'All_Beauty', price: 89000, rating: 4.3, review_count: 156, description: 'Mặt nạ collagen kết hợp vàng 24K giúp da căng bóng, giảm nếp nhăn sau một lần sử dụng.', image_url: 'https://picsum.photos/seed/beauty4/400/400' },
  { id: 'b005', title: 'Nước Tẩy Trang Micellar Water 500ml', category: 'All_Beauty', price: 215000, rating: 4.6, review_count: 423, description: 'Nước tẩy trang dịu nhẹ, làm sạch sâu lớp trang điểm mà không gây khô da. Phù hợp da nhạy cảm.', image_url: 'https://picsum.photos/seed/beauty5/400/400' },
  { id: 'b006', title: 'Phấn Phủ Kiềm Dầu Siêu Mịn', category: 'All_Beauty', price: 275000, rating: 4.4, review_count: 312, description: 'Phấn phủ kiềm dầu 24h, giúp da luôn tươi matte. Texture siêu mịn, không lộ lỗ chân lông.', image_url: 'https://picsum.photos/seed/beauty6/400/400' },
  { id: 'b007', title: 'Bộ Cọ Trang Điểm 12 Cây Cao Cấp', category: 'All_Beauty', price: 650000, rating: 4.9, review_count: 89, description: 'Bộ cọ trang điểm chuyên nghiệp 12 cây với lông cọ siêu mềm, cán gỗ sang trọng. Đầy đủ cho mọi bước makeup.', image_url: 'https://picsum.photos/seed/beauty7/400/400' },
  { id: 'b008', title: 'Dầu Gội Dược Liệu Ngăn Rụng Tóc', category: 'All_Beauty', price: 195000, rating: 4.2, review_count: 278, description: 'Dầu gội thảo dược giúp giảm rụng tóc, kích thích mọc tóc mới. Thành phần tự nhiên 100%.', image_url: 'https://picsum.photos/seed/beauty8/400/400' },
  { id: 'b009', title: 'Cushion Foundation Che Phủ Hoàn Hảo', category: 'All_Beauty', price: 520000, rating: 4.6, review_count: 445, description: 'Cushion che phủ tốt với finish bán lì tự nhiên. SPF 50+ bảo vệ da suốt ngày dài.', image_url: 'https://picsum.photos/seed/beauty9/400/400' },
  { id: 'b010', title: 'Xịt Khoáng Cấp Ẩm Tức Thì', category: 'All_Beauty', price: 145000, rating: 4.1, review_count: 198, description: 'Xịt khoáng từ suối nước nóng thiên nhiên, cấp ẩm và làm dịu da tức thì mọi lúc mọi nơi.', image_url: 'https://picsum.photos/seed/beauty10/400/400' },

  // ── Video_Games (11) ──
  { id: 'g001', title: 'PlayStation 5 DualSense Controller', category: 'Video_Games', price: 1850000, rating: 4.9, review_count: 1234, description: 'Tay cầm PS5 DualSense với haptic feedback và adaptive triggers, mang đến trải nghiệm chơi game chân thực nhất.', image_url: 'https://picsum.photos/seed/game1/400/400' },
  { id: 'g002', title: 'Nintendo Switch OLED Edition', category: 'Video_Games', price: 8500000, rating: 4.8, review_count: 2341, description: 'Nintendo Switch phiên bản OLED với màn hình 7 inch sắc nét. Chơi mọi lúc mọi nơi.', image_url: 'https://picsum.photos/seed/game2/400/400' },
  { id: 'g003', title: 'Tai Nghe Gaming RGB 7.1 Surround', category: 'Video_Games', price: 950000, rating: 4.5, review_count: 678, description: 'Tai nghe gaming với âm thanh vòm 7.1, đèn RGB cùng micro khử ồn chuyên nghiệp.', image_url: 'https://picsum.photos/seed/game3/400/400' },
  { id: 'g004', title: 'Bàn Phím Cơ Gaming Cherry MX', category: 'Video_Games', price: 2200000, rating: 4.7, review_count: 456, description: 'Bàn phím cơ với switch Cherry MX Blue, full RGB, khung nhôm CNC chắc chắn.', image_url: 'https://picsum.photos/seed/game4/400/400' },
  { id: 'g005', title: 'Chuột Gaming Không Dây 25600 DPI', category: 'Video_Games', price: 1450000, rating: 4.6, review_count: 789, description: 'Chuột gaming không dây siêu nhẹ 63g, sensor 25600 DPI, pin 70h. Hoàn hảo cho FPS.', image_url: 'https://picsum.photos/seed/game5/400/400' },
  { id: 'g006', title: 'Steam Deck 512GB Console', category: 'Video_Games', price: 14500000, rating: 4.4, review_count: 345, description: 'Máy chơi game cầm tay Steam Deck 512GB, chơi được toàn bộ thư viện Steam ở mọi nơi.', image_url: 'https://picsum.photos/seed/game6/400/400' },
  { id: 'g007', title: 'Ghế Gaming Ergonomic Pro', category: 'Video_Games', price: 4200000, rating: 4.3, review_count: 234, description: 'Ghế gaming cao cấp với tựa lưng ergonomic, đệm memory foam, nâng hạ khí nén.', image_url: 'https://picsum.photos/seed/game7/400/400' },
  { id: 'g008', title: 'Tay Cầm Xbox Wireless Đặc Biệt', category: 'Video_Games', price: 1650000, rating: 4.7, review_count: 567, description: 'Tay cầm Xbox Elite Series 2 Core phiên bản đặc biệt, textured grip chống trơn.', image_url: 'https://picsum.photos/seed/game8/400/400' },
  { id: 'g009', title: 'Màn Hình Gaming 27" 165Hz IPS', category: 'Video_Games', price: 6800000, rating: 4.8, review_count: 912, description: 'Màn hình gaming 27" QHD 165Hz IPS với thời gian phản hồi 1ms, G-Sync compatible.', image_url: 'https://picsum.photos/seed/game9/400/400' },
  { id: 'g010', title: 'Bàn Di Chuột XXL RGB', category: 'Video_Games', price: 380000, rating: 4.2, review_count: 423, description: 'Bàn di chuột kích thước XXL 900x400mm với viền LED RGB, bề mặt micro-texture chính xác.', image_url: 'https://picsum.photos/seed/game10/400/400' },
  { id: 'g011', title: 'Capture Card 4K60 Streaming', category: 'Video_Games', price: 3200000, rating: 4.5, review_count: 156, description: 'Card capture 4K60 HDR passthrough, ghi hình 1080p60, tương thích OBS và mọi phần mềm streaming.', image_url: 'https://picsum.photos/seed/game11/400/400' },

  // ── Amazon_Fashion (10) ──
  { id: 'f001', title: 'Áo Thun Oversize Basic Cotton', category: 'Amazon_Fashion', price: 250000, rating: 4.4, review_count: 1567, description: 'Áo thun oversize chất cotton 100% mềm mịn, form rộng thoải mái. Unisex, nhiều màu sắc.', image_url: 'https://picsum.photos/seed/fashion1/400/400' },
  { id: 'f002', title: 'Quần Jogger Streetwear Cao Cấp', category: 'Amazon_Fashion', price: 450000, rating: 4.5, review_count: 890, description: 'Quần jogger phong cách streetwear, chất vải tech-fleece co giãn 4 chiều, ống bo thời trang.', image_url: 'https://picsum.photos/seed/fashion2/400/400' },
  { id: 'f003', title: 'Giày Sneaker Retro Classic White', category: 'Amazon_Fashion', price: 1250000, rating: 4.7, review_count: 2345, description: 'Giày sneaker retro trắng tinh khôi, đế cao su bền bỉ, phối đồ dễ dàng mọi phong cách.', image_url: 'https://picsum.photos/seed/fashion3/400/400' },
  { id: 'f004', title: 'Kính Mát Phân Cực UV400', category: 'Amazon_Fashion', price: 380000, rating: 4.3, review_count: 567, description: 'Kính mát phân cực chống UV400, khung titanium siêu nhẹ, phong cách thời thượng.', image_url: 'https://picsum.photos/seed/fashion4/400/400' },
  { id: 'f005', title: 'Túi Tote Canvas Minimalist', category: 'Amazon_Fashion', price: 195000, rating: 4.1, review_count: 345, description: 'Túi tote canvas thiết kế tối giản, vải dày chắc chắn, phù hợp đi học và đi làm.', image_url: 'https://picsum.photos/seed/fashion5/400/400' },
  { id: 'f006', title: 'Áo Khoác Bomber Jacket Unisex', category: 'Amazon_Fashion', price: 680000, rating: 4.6, review_count: 423, description: 'Áo khoác bomber style Nhật Bản, lót lụa mềm mại, thêu logo tinh tế ở lưng.', image_url: 'https://picsum.photos/seed/fashion6/400/400' },
  { id: 'f007', title: 'Đồng Hồ Minimalist Rose Gold', category: 'Amazon_Fashion', price: 1850000, rating: 4.8, review_count: 678, description: 'Đồng hồ thiết kế tối giản mặt 38mm, dây da Italy, máy Nhật Bản. Phong cách thanh lịch.', image_url: 'https://picsum.photos/seed/fashion7/400/400' },
  { id: 'f008', title: 'Nón Bucket Hat Unisex', category: 'Amazon_Fashion', price: 165000, rating: 4.2, review_count: 789, description: 'Nón bucket hat vải cotton wash, phong cách Y2K trendy, chống nắng nhẹ nhàng.', image_url: 'https://picsum.photos/seed/fashion8/400/400' },
  { id: 'f009', title: 'Balo Laptop Chống Nước 15.6"', category: 'Amazon_Fashion', price: 520000, rating: 4.5, review_count: 456, description: 'Balo laptop chống nước chất liệu Oxford, ngăn laptop 15.6", cổng sạc USB tích hợp.', image_url: 'https://picsum.photos/seed/fashion9/400/400' },
  { id: 'f010', title: 'Váy Midi Floral Mùa Hè', category: 'Amazon_Fashion', price: 420000, rating: 4.4, review_count: 312, description: 'Váy midi họa tiết hoa vintage, chất vải chiffon bay bổng, hoàn hảo cho mùa hè.', image_url: 'https://picsum.photos/seed/fashion10/400/400' },

  // ── Baby_Products (11) ──
  { id: 'p001', title: 'Bình Sữa Cho Bé 240ml Anti-Colic', category: 'Baby_Products', price: 285000, rating: 4.8, review_count: 1234, description: 'Bình sữa chống đầy hơi cho bé, chất liệu PPSU an toàn, núm ty silicone mềm tự nhiên.', image_url: 'https://picsum.photos/seed/baby1/400/400' },
  { id: 'p002', title: 'Tã Dán Sơ Sinh Siêu Thấm Size S', category: 'Baby_Products', price: 320000, rating: 4.6, review_count: 2567, description: 'Tã dán cho bé sơ sinh, siêu thấm 12h, mỏng nhẹ thoáng khí, không gây hăm tã.', image_url: 'https://picsum.photos/seed/baby2/400/400' },
  { id: 'p003', title: 'Xe Đẩy Gấp Gọn Siêu Nhẹ 5.5kg', category: 'Baby_Products', price: 3500000, rating: 4.7, review_count: 456, description: 'Xe đẩy em bé siêu nhẹ 5.5kg, gấp gọn 1 tay, giảm xóc 4 bánh, mái che UV50+.', image_url: 'https://picsum.photos/seed/baby3/400/400' },
  { id: 'p004', title: 'Bộ Đồ Chơi Xếp Hình Montessori', category: 'Baby_Products', price: 195000, rating: 4.5, review_count: 789, description: 'Bộ đồ chơi xếp hình gỗ Montessori cho bé 1-3 tuổi, sơn an toàn, phát triển tư duy.', image_url: 'https://picsum.photos/seed/baby4/400/400' },
  { id: 'p005', title: 'Sữa Tắm Gội Toàn Thân Cho Bé', category: 'Baby_Products', price: 145000, rating: 4.4, review_count: 1678, description: 'Sữa tắm gội 2 in 1 cho bé, công thức dịu nhẹ "không nước mắt", pH 5.5 cân bằng.', image_url: 'https://picsum.photos/seed/baby5/400/400' },
  { id: 'p006', title: 'Ghế Ăn Dặm Đa Năng 3 in 1', category: 'Baby_Products', price: 1850000, rating: 4.6, review_count: 345, description: 'Ghế ăn dặm cho bé 3 in 1: ghế cao, ghế thấp và booster seat. Khay ăn có thể tháo rời.', image_url: 'https://picsum.photos/seed/baby6/400/400' },
  { id: 'p007', title: 'Máy Hút Sữa Điện Đôi Thông Minh', category: 'Baby_Products', price: 2450000, rating: 4.9, review_count: 234, description: 'Máy hút sữa điện đôi với 9 mức hút, chế độ massage kích sữa, motor siêu êm.', image_url: 'https://picsum.photos/seed/baby7/400/400' },
  { id: 'p008', title: 'Yếm Ăn Silicon Chống Thấm', category: 'Baby_Products', price: 89000, rating: 4.3, review_count: 567, description: 'Yếm ăn dặm silicon food-grade, có máng hứng, dễ vệ sinh, nhiều màu dễ thương.', image_url: 'https://picsum.photos/seed/baby8/400/400' },
  { id: 'p009', title: 'Đèn Ngủ Chiếu Sao Cho Bé', category: 'Baby_Products', price: 250000, rating: 4.2, review_count: 423, description: 'Đèn ngủ chiếu sao xoay 360 độ, 6 chế độ màu, nhạc ru ngủ, hẹn giờ tự tắt.', image_url: 'https://picsum.photos/seed/baby9/400/400' },
  { id: 'p010', title: 'Bộ Chén Bát Ăn Dặm Lúa Mì', category: 'Baby_Products', price: 125000, rating: 4.1, review_count: 890, description: 'Bộ bát đĩa ăn dặm từ sợi lúa mì tự nhiên, chống vỡ, an toàn cho bé, bao gồm 5 món.', image_url: 'https://picsum.photos/seed/baby10/400/400' },
  { id: 'p011', title: 'Nôi Điện Tự Đưa Thông Minh', category: 'Baby_Products', price: 1950000, rating: 4.7, review_count: 178, description: 'Nôi điện thông minh tự phát hiện bé khóc và đưa nôi, 5 tốc độ, nhạc ru tích hợp.', image_url: 'https://picsum.photos/seed/baby11/400/400' }
];

// ─── HELPERS ─────────────────────────────────────────
function formatPrice(price) {
  return new Intl.NumberFormat('vi-VN').format(price) + '₫';
}

function generateStars(rating) {
  let html = '<div class="stars">';
  for (let i = 1; i <= 5; i++) {
    if (i <= Math.floor(rating)) {
      html += '<span class="star filled">★</span>';
    } else if (i - 0.5 <= rating) {
      html += '<span class="star half">★</span>';
    } else {
      html += '<span class="star">★</span>';
    }
  }
  html += '</div>';
  return html;
}

function getCategoryInfo(categoryId) {
  return CATEGORIES.find(c => c.id === categoryId) || CATEGORIES[0];
}

function getCategoryBadgeClass(categoryId) {
  const map = {
    'All_Beauty': 'badge-beauty',
    'Video_Games': 'badge-games',
    'Amazon_Fashion': 'badge-fashion',
    'Baby_Products': 'badge-baby'
  };
  return map[categoryId] || '';
}

function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function escapeHtml(text) {
  if (text == null) return '';
  const d = document.createElement('div');
  d.textContent = String(text);
  return d.innerHTML;
}

function escapeAttr(text) {
  return escapeHtml(text).replace(/"/g, '&quot;');
}

/** URL ảnh ổn định + fallback SVG khi CDN lỗi */
function productImageSrc(product) {
  const url = (product && product.image_url) ? String(product.image_url).trim() : '';
  if (url.startsWith('http') || url.startsWith('data:')) return url;
  const cat = product?.category || 'All_Beauty';
  const id = product?.id || product?.item_id || 'item';
  return `https://picsum.photos/seed/fedshop_${cat}_${id}/400/400`;
}

function productImageFallback(product) {
  const fb = product?.image_fallback;
  if (fb && fb.startsWith('data:')) return fb;
  const cat = product?.category || 'All_Beauty';
  const colors = {
    All_Beauty: 'fd79a8',
    Video_Games: '6c5ce7',
    Amazon_Fashion: '00cec9',
    Baby_Products: 'fdcb6e'
  };
  const c = colors[cat] || '6c5ce7';
  const label = (cat.split('_')[1] || 'Shop').slice(0, 8);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400"><rect width="400" height="400" fill="#${c}"/><text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" fill="#fff" font-size="20">${label}</text></svg>`;
  return 'data:image/svg+xml,' + encodeURIComponent(svg);
}

function onProductImgError(img, category) {
  if (!img || img.dataset.fallbackApplied === '1') return;
  img.dataset.fallbackApplied = '1';
  img.src = productImageFallback({ category: category || 'All_Beauty' });
}

function productImgTag(product, extraClass = '') {
  const src = escapeAttr(productImageSrc(product));
  const cat = escapeAttr(product.category || 'All_Beauty');
  return `<img class="${extraClass}" src="${src}" alt="${escapeAttr(product.title)}" loading="lazy" ` +
    `onerror="onProductImgError(this,'${cat}')">`;
}

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).substring(2, 15);
}

// ─── AUTH (Federated Client Login) ───────────────────
class AuthManager {
  constructor(api) {
    this.api = api;
    this.session = null;
  }

  loadStored() {
    try {
      const raw = localStorage.getItem(SESSION_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  save(session) {
    this.session = session;
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
    this.api.setAuthSession(session);
  }

  clear() {
    this.session = null;
    localStorage.removeItem(SESSION_STORAGE_KEY);
    this.api.setAuthSession(null);
  }

  async fetchClients() {
    const data = await this.api._fetch(`${API_BASE}/api/auth/clients`);
    return data?.clients || [];
  }

  async fetchUsers(clientId) {
    const data = await this.api._fetch(`${API_BASE}/api/auth/clients/${clientId}/users`);
    return data?.users || [];
  }

  async login(clientId, userId) {
    const data = await this.api._fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId, user_id: userId })
    });
    if (!data?.session_id) return null;
    this.save(data);
    return data;
  }
}

// ─── API CLIENT ──────────────────────────────────────
class APIClient {
  constructor() {
    this.isOffline = false;
    this.sessionId = generateSessionId();
    this.authSession = null;
  }

  setAuthSession(session) {
    this.authSession = session;
    if (session?.session_id) {
      this.sessionId = session.session_id;
    }
  }

  async _fetch(url, options = {}) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeout);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.isOffline = false;
      return await response.json();
    } catch (err) {
      this.isOffline = true;
      return null;
    }
  }

  async fetchProducts(category = null, page = 1, sort = 'default') {
    const params = new URLSearchParams({ page: String(page), sort, limit: '80' });
    if (category) params.set('category', category);
    if (this.sessionId) params.set('session_id', this.sessionId);
    if (this.authSession?.client_id != null) {
      params.set('client_id', String(this.authSession.client_id));
    }
    const data = await this._fetch(`${API_BASE}/api/products?${params}`);
    if (data) return data;

    // Fallback to mock
    let products = [...MOCK_PRODUCTS];
    if (category) products = products.filter(p => p.category === category);
    return { products, total: products.length, page: 1 };
  }

  async fetchRecommendations(sessionId, productContext = null) {
    const body = { session_id: sessionId, top_k: 8 };
    if (productContext) body.context_item_id = productContext;
    if (this.authSession?.client_id != null) {
      body.client_id = this.authSession.client_id;
      body.category = this.authSession.category;
    }
    const data = await this._fetch(`${API_BASE}/api/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (data) return data;

    // Fallback: shuffle and pick 8
    const shuffled = [...MOCK_PRODUCTS].sort(() => Math.random() - 0.5);
    return {
      recommendations: shuffled.slice(0, 8),
      fusion_weights: { text: 0.45, image: 0.30, behavior: 0.25 }
    };
  }

  async trackBehavior(productId, action, duration = null, extra = {}) {
    const body = {
      session_id: this.sessionId,
      product_id: productId,
      item_id: productId,
      action,
      event_type: action,
      duration,
      timestamp: new Date().toISOString(),
      ...extra
    };
    if (this.authSession?.client_id != null) {
      body.client_id = this.authSession.client_id;
    }
    return await this._fetch(`${API_BASE}/api/track-behavior`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  }

  async searchProducts(query) {
    const data = await this._fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`);
    if (data) return data;

    // Fallback: local search
    const q = query.toLowerCase();
    const results = MOCK_PRODUCTS.filter(p =>
      p.title.toLowerCase().includes(q) ||
      p.description.toLowerCase().includes(q) ||
      p.category.toLowerCase().includes(q)
    );
    return { products: results, total: results.length };
  }
}

// ─── ACTIVITY LOG PANEL ─────────────────────────────
const ACTION_LABELS = {
  click: '👆 Click xem',
  view: '👁️ Xem chi tiết',
  add_to_cart: '🛒 Thêm giỏ',
  search: '🔍 Tìm kiếm',
  login: '🔐 Đăng nhập',
  recommend: '✨ Làm mới gợi ý',
  batch: '📦 Gửi batch'
};

class ActivityLogPanel {
  constructor() {
    this.panel = document.getElementById('activityPanel');
    this.list = document.getElementById('activityLogList');
    this.foot = document.getElementById('activityPanelFoot');
    this.badge = document.getElementById('activityApiBadge');
    this.maxEntries = 14;
    this._pendingId = 0;

    document.getElementById('activityPanelToggle')?.addEventListener('click', () => {
      this.panel?.classList.toggle('collapsed');
      const btn = document.getElementById('activityPanelToggle');
      if (btn) btn.textContent = this.panel?.classList.contains('collapsed') ? '+' : '−';
    });
  }

  setApiOnline(online) {
    if (!this.badge) return;
    this.badge.textContent = online ? 'API online' : 'Offline';
    this.badge.classList.toggle('offline', !online);
    if (this.foot) {
      this.foot.textContent = online
        ? 'POST /api/track-behavior → behavior vector + FedPer personal head'
        : 'Offline — sự kiện chỉ lưu trên trình duyệt';
    }
  }

  show() {
    this.panel?.removeAttribute('hidden');
  }

  hide() {
    this.panel?.setAttribute('hidden', '');
  }

  _timeStr() {
    return new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  _escape(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  push(entry) {
    if (!this.list) return;

    const empty = this.list.querySelector('.activity-empty');
    if (empty) empty.remove();

    const id = `act-${++this._pendingId}`;
    const el = document.createElement('div');
    el.className = `activity-entry ${entry.status || 'ok'}`;
    el.id = id;
    el.innerHTML = `
      <div class="activity-entry-top">
        <span class="activity-action">${this._escape(entry.actionLabel)}</span>
        <span class="activity-time">${entry.time}</span>
      </div>
      ${entry.productLine ? `<div class="activity-product">${this._escape(entry.productLine)}</div>` : ''}
      <div class="activity-response">${this._escape(entry.responseLine)}</div>
    `;
    this.list.prepend(el);

    while (this.list.children.length > this.maxEntries) {
      this.list.lastElementChild?.remove();
    }
    return id;
  }

  pushPending(action, productId, productTitle) {
    const label = ACTION_LABELS[action] || action;
    const productLine = productId
      ? `${productTitle || productId}`
      : '—';
    return this.push({
      status: 'pending',
      actionLabel: label,
      time: this._timeStr(),
      productLine,
      responseLine: 'Đang gửi → POST /api/track-behavior …'
    });
  }

  complete(entryId, action, productId, productTitle, apiRes, apiOnline) {
    const el = entryId ? document.getElementById(entryId) : null;
    const label = ACTION_LABELS[action] || action;
    let status = 'ok';
    let responseLine = '';

    if (!apiOnline || apiRes === null) {
      status = 'offline';
      responseLine = '⚠ Không gửi server (offline / API tắt)';
    } else if (apiRes.status === 'ok') {
      const fl = apiRes.fl_update || {};
      if (fl.updated) {
        status = 'fl-updated';
        responseLine =
          `✓ Server OK · FL personal head cập nhật · mẫu=${fl.samples_used} · loss=${fl.loss} · lần #${fl.personal_updates || 1}`;
      } else {
        responseLine =
          `✓ Server OK · buffer=${fl.buffer_size ?? '—'} · ${fl.reason === 'not_enough_interactions' ? 'cần thêm tương tác để học' : 'chờ đủ mẫu'}`;
      }
    } else {
      status = 'error';
      responseLine = '✗ Phản hồi API lỗi';
    }

    if (el) {
      el.className = `activity-entry ${status}`;
      const resp = el.querySelector('.activity-response');
      if (resp) resp.textContent = responseLine;
    } else {
      this.push({
        status,
        actionLabel: label,
        time: this._timeStr(),
        productLine: productId ? (productTitle || productId) : '—',
        responseLine
      });
    }
  }

  logInfo(action, message, productLine = '') {
    this.push({
      status: 'ok',
      actionLabel: ACTION_LABELS[action] || action,
      time: this._timeStr(),
      productLine,
      responseLine: message
    });
  }
}

// ─── BEHAVIOR TRACKER ────────────────────────────────
class BehaviorTracker {
  constructor(apiClient) {
    this.api = apiClient;
    this.clicks = {};
    this.views = {};
    this.cartAdds = {};
    this.searchQueries = [];
    this._startAutoSend();
  }

  async recordClick(productId) {
    this.clicks[productId] = (this.clicks[productId] || 0) + 1;
    return this.api.trackBehavior(productId, 'click');
  }

  async recordView(productId, startTime) {
    const duration = Date.now() - startTime;
    this.views[productId] = (this.views[productId] || 0) + duration;
    return this.api.trackBehavior(productId, 'view', duration);
  }

  async recordAddToCart(productId) {
    this.cartAdds[productId] = (this.cartAdds[productId] || 0) + 1;
    return this.api.trackBehavior(productId, 'add_to_cart');
  }

  recordSearch(query) {
    this.searchQueries.push({ query, timestamp: Date.now() });
    this.api.trackBehavior(null, 'search');
  }

  getSessionBehavior() {
    return {
      clicks: { ...this.clicks },
      views: { ...this.views },
      cart_adds: { ...this.cartAdds },
      search_queries: this.searchQueries.slice(-10)
    };
  }

  _startAutoSend() {
    setInterval(() => {
      const behavior = this.getSessionBehavior();
      if (Object.keys(behavior.clicks).length > 0 || Object.keys(behavior.views).length > 0) {
        this.api._fetch(`${API_BASE}/api/track-behavior/batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: this.api.sessionId,
            behavior
          })
        });
      }
    }, 30000);
  }
}

// ─── CART ────────────────────────────────────────────
class Cart {
  constructor() {
    this.items = this._load();
    this._updateBadge();
  }

  _load() {
    try {
      return JSON.parse(localStorage.getItem('fedshop_cart')) || [];
    } catch {
      return [];
    }
  }

  _save() {
    localStorage.setItem('fedshop_cart', JSON.stringify(this.items));
    this._updateBadge();
  }

  _updateBadge() {
    const badge = document.getElementById('cartBadge');
    if (!badge) return;
    const count = this.getCount();
    badge.textContent = count;
    badge.classList.toggle('show', count > 0);
  }

  add(product) {
    const existing = this.items.find(i => i.id === product.id);
    if (existing) {
      existing.qty += 1;
    } else {
      this.items.push({ ...product, qty: 1 });
    }
    this._save();
  }

  remove(productId) {
    this.items = this.items.filter(i => i.id !== productId);
    this._save();
  }

  updateQty(productId, delta) {
    const item = this.items.find(i => i.id === productId);
    if (!item) return;
    item.qty += delta;
    if (item.qty <= 0) {
      this.remove(productId);
    } else {
      this._save();
    }
  }

  getTotal() {
    return this.items.reduce((sum, i) => sum + i.price * i.qty, 0);
  }

  getCount() {
    return this.items.reduce((sum, i) => sum + i.qty, 0);
  }

  renderCart() {
    const container = document.getElementById('cartItems');
    const totalEl = document.getElementById('cartTotal');
    if (!container || !totalEl) return;

    if (this.items.length === 0) {
      container.innerHTML = `
        <div class="cart-empty">
          <div class="empty-icon">🛒</div>
          <p>Giỏ hàng trống</p>
          <p style="font-size: 13px; margin-top: 8px;">Thêm sản phẩm để bắt đầu mua sắm!</p>
        </div>`;
    } else {
      container.innerHTML = this.items.map(item => `
        <div class="cart-item" data-id="${escapeAttr(item.id)}">
          <div class="cart-item-image">
            ${productImgTag(item)}
          </div>
          <div class="cart-item-info">
            <div class="cart-item-title">${escapeHtml(item.title)}</div>
            <div class="cart-item-price">${formatPrice(item.price)}</div>
            <div class="cart-item-qty">
              <button type="button" data-cart-action="minus" data-product-id="${escapeAttr(item.id)}">−</button>
              <span>${item.qty}</span>
              <button type="button" data-cart-action="plus" data-product-id="${escapeAttr(item.id)}">+</button>
            </div>
          </div>
          <button type="button" class="cart-item-remove" data-cart-action="remove" data-product-id="${escapeAttr(item.id)}" aria-label="Xóa">✕</button>
        </div>
      `).join('');
    }

    totalEl.textContent = formatPrice(this.getTotal());
  }
}

// ─── RECOMMENDATION WIDGET ──────────────────────────
class RecommendationWidget {
  constructor(apiClient) {
    this.api = apiClient;
    this.container = document.getElementById('recProducts');
    this.refreshInterval = null;
  }

  async loadRecommendations(productContext = null) {
    if (!this.container) return;

    // Show skeletons
    this.container.innerHTML = Array(4).fill(`
      <div class="product-card skeleton" style="min-width:260px; max-width:280px; flex-shrink:0;">
        <div class="skeleton-image"></div>
        <div class="skeleton-line" style="width:80%"></div>
        <div class="skeleton-line short"></div>
        <div class="skeleton-line price"></div>
      </div>
    `).join('');

    const data = await this.api.fetchRecommendations(this.api.sessionId, productContext);
    if (data) {
      this.renderRecommendations(data.recommendations, data.fusion_weights);
      if (window.app && data.fl_stats) {
        window.app._updateFlStatus({ fl_stats: data.fl_stats });
      }
    }
  }

  renderRecommendations(products, fusionWeights) {
    if (!this.container) return;

    this.container.innerHTML = products.map((product, i) => `
      <div class="product-card" data-product-id="${escapeAttr(product.id)}" style="min-width:260px; max-width:280px; flex-shrink:0; animation-delay:${i * 0.08}s; cursor:pointer">
        <div class="product-card-image">
          ${productImgTag(product)}
        </div>
        <div class="product-card-body">
          <span class="product-card-category">${escapeHtml(getCategoryInfo(product.category).name)}</span>
          <h3 class="product-card-title">${escapeHtml(product.title)}</h3>
          <div class="product-card-rating">
            ${generateStars(product.rating)}
            <span class="review-count">(${product.review_count})</span>
          </div>
          <div class="product-card-price">
            <span class="price-current">${formatPrice(product.price)}</span>
          </div>
          <div class="product-card-actions">
            <button type="button" class="btn-add-cart" data-action="cart">🛒 Thêm</button>
            <button type="button" class="btn-quick-view" data-action="view">👁️</button>
          </div>
        </div>
      </div>
    `).join('');

    if (fusionWeights) {
      this.updateFusionWeights(fusionWeights);
    }
  }

  updateFusionWeights(weights) {
    const textPct = Math.round((weights.text || 0.45) * 100);
    const imagePct = Math.round((weights.image || 0.30) * 100);
    const behaviorPct = Math.round((weights.behavior || 0.25) * 100);

    const textVal = document.getElementById('fusionText');
    const imageVal = document.getElementById('fusionImage');
    const behaviorVal = document.getElementById('fusionBehavior');
    const textBar = document.getElementById('fusionTextBar');
    const imageBar = document.getElementById('fusionImageBar');
    const behaviorBar = document.getElementById('fusionBehaviorBar');

    if (textVal) textVal.textContent = textPct + '%';
    if (imageVal) imageVal.textContent = imagePct + '%';
    if (behaviorVal) behaviorVal.textContent = behaviorPct + '%';

    // Animate bars
    requestAnimationFrame(() => {
      if (textBar) textBar.style.width = textPct + '%';
      if (imageBar) imageBar.style.width = imagePct + '%';
      if (behaviorBar) behaviorBar.style.width = behaviorPct + '%';
    });
  }

  startAutoRefresh() {
    this.refreshInterval = setInterval(() => {
      this.loadRecommendations();
    }, 60000);
  }

  stopAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
  }
}

// ─── APP (Main Controller) ──────────────────────────
class App {
  constructor() {
    this.api = new APIClient();
    this.auth = new AuthManager(this.api);
    this.tracker = new BehaviorTracker(this.api);
    this.cart = new Cart();
    this.recWidget = new RecommendationWidget(this.api);
    this.activityLog = new ActivityLogPanel();

    this.allProducts = [...MOCK_PRODUCTS];
    this.filteredProducts = [...MOCK_PRODUCTS];
    this.currentCategory = null;
    this.currentSort = 'default';
    this.modalViewStart = null;
    this.currentModalProductId = null;
    this.clients = [];

    this._init();
  }

  async _init() {
    this._bindEvents();
    this._setupScrollAnimations();
    this._setupNavbarScroll();

    const stored = this.auth.loadStored();
    if (stored?.session_id) {
      this.auth.save(stored);
      await this._enterShop(stored);
    } else {
      await this._showLogin();
    }
  }

  _buildOfflineSession(clientId, userId) {
    const meta = STATIC_DEMO_CLIENTS.find(c => c.client_id === clientId) || STATIC_DEMO_CLIENTS[0];
    const products = MOCK_PRODUCTS.filter(p => p.category === meta.category);
    return {
      session_id: `offline_c${clientId}_${Date.now()}`,
      client_id: clientId,
      user_id: userId || `demo_user_${clientId}`,
      category: meta.category,
      store_name: meta.store_name,
      user_display: `Khách demo #${(clientId % 10) + 1}`,
      products: products.length ? products : [...MOCK_PRODUCTS],
      offline_mode: true
    };
  }

  _fillClientSelect(clientSelect, clients) {
    clientSelect.innerHTML = clients.map(c =>
      `<option value="${c.client_id}">${c.store_name} (${c.category_vi})</option>`
    ).join('');
  }

  async _showLogin() {
    const overlay = document.getElementById('loginOverlay');
    const clientSelect = document.getElementById('clientSelect');
    const userSelect = document.getElementById('userSelect');
    const hintEl = document.getElementById('loginHint');
    overlay?.classList.add('active');
    document.getElementById('flStatusBar')?.setAttribute('hidden', '');

    const apiClients = await this.auth.fetchClients();
    const apiOnline = apiClients.length > 0;
    this.clients = apiOnline ? apiClients : STATIC_DEMO_CLIENTS;
    this._fillClientSelect(clientSelect, this.clients);

    if (!apiOnline) {
      hintEl.textContent =
        'API chưa chạy — chọn client bên dưới rồi bấm «Vào shop ngay» hoặc khởi động: python src/api/fastapi_app.py';
      this.api.isOffline = true;
    }

    const onClientChange = async () => {
      const cid = parseInt(clientSelect.value, 10);
      if (Number.isNaN(cid)) return;

      let users = [];
      if (apiOnline) {
        users = await this.auth.fetchUsers(cid);
      }
      if (!users.length) {
        users = Array.from({ length: 5 }, (_, i) => ({
          user_id: `demo_user_${cid}_${i}`,
          display_name: `Khách hàng #${i + 1}`,
          interactions: 15 - i
        }));
      }
      userSelect.innerHTML = users.map(u =>
        `<option value="${u.user_id}">${u.display_name} — ${u.interactions} tương tác</option>`
      ).join('');
      hintEl.textContent = apiOnline
        ? `Client #${cid}: ${users.length} khách hàng trong silo FL.`
        : `Chế độ offline — Client #${cid}, ${users.length} khách mẫu (dữ liệu mock).`;
    };

    clientSelect.onchange = onClientChange;
    await onClientChange();

    const finishLogin = async (useOffline) => {
      const cid = parseInt(clientSelect.value, 10);
      const uid = userSelect.value;
      let sess = null;

      if (!useOffline && apiOnline) {
        sess = await this.auth.login(cid, uid);
      }
      if (!sess) {
        sess = this._buildOfflineSession(cid, uid);
        this.auth.save(sess);
        this.api.isOffline = true;
        this.showToast('Chế độ offline — dùng dữ liệu mẫu trên trình duyệt', 'info');
      }

      this._hideLoginOverlay();
      await this._enterShop(sess);
    };

    document.getElementById('btnLogin').onclick = () => finishLogin(false);
    document.getElementById('btnOffline').onclick = () => finishLogin(true);
  }

  _hideLoginOverlay() {
    const overlay = document.getElementById('loginOverlay');
    if (!overlay) return;
    overlay.classList.remove('active');
    overlay.style.display = 'none';
  }

  _revealAllSections() {
    document.querySelectorAll('.section').forEach((el) => {
      el.classList.add('visible');
    });
  }

  async _enterShop(session) {
    this._hideLoginOverlay();
    this._revealAllSections();
    this.activityLog.show();
    this.activityLog.setApiOnline(!session.offline_mode && !this.api.isOffline);
    this.activityLog.logInfo(
      'login',
      `Session ${session.session_id?.slice(0, 20)}… · Client #${session.client_id} · ${session.offline_mode ? 'chế độ offline' : 'API + FedPer'}`,
      session.store_name || ''
    );
    document.body.classList.add('has-fl-bar');
    const bar = document.getElementById('flStatusBar');
    bar?.removeAttribute('hidden');

    const offline = !!session.offline_mode;
    document.getElementById('flUserLabel').textContent =
      `${session.store_name} · ${session.user_display || session.user_id}${offline ? ' (offline)' : ''}`;
    document.getElementById('heroSubtitle').textContent =
      offline
        ? `Demo offline tại ${session.store_name}. Bật API để dùng mô hình FedPer thật.`
        : `Bạn đang mua sắm tại ${session.store_name}. Gợi ý từ Client #${session.client_id} — cập nhật khi bạn tương tác.`;

    this.currentCategory = session.category || null;
    this._renderCategories();

    if (session.products?.length) {
      this.allProducts = session.products;
      this.filteredProducts = [...session.products];
      this._sortProducts();
      this._renderProducts();
    } else {
      await this.loadProducts(this.currentCategory);
    }

    await this.recWidget.loadRecommendations();
    this.recWidget.startAutoRefresh();
    this._updateConnectionStatus();
    this.activityLog.setApiOnline(!this.api.isOffline && !session.offline_mode);

    await this._loadModelBadge();
  }

  async _loadModelBadge() {
    const badge = document.getElementById('modelBadge');
    if (!badge) return;
    const data = await this.api._fetch(`${API_BASE}/api/model-info`);
    if (!data?.ok) {
      badge.textContent = 'Model: offline';
      return;
    }
    const parts = [];
    if (data.architecture && data.architecture !== 'unloaded') parts.push(data.architecture);
    if (data.text_model) parts.push(`text=${data.text_model}`);
    if (data.image_model) parts.push(`img=${data.image_model}`);
    badge.textContent = `Model: ${parts.join(' · ') || data.architecture || '—'}`;
    badge.title = (data.notes || '').trim() || badge.textContent;
  }

  _updateFlStatus(flPayload) {
    const el = document.getElementById('flUpdateLabel');
    const bar = document.getElementById('flStatusBar');
    if (!el) return;
    if (flPayload?.updated) {
      el.textContent = `✨ Personal head cập nhật (#${flPayload.personal_updates || 1}, loss ${flPayload.loss})`;
      bar?.classList.add('fl-pulse');
      setTimeout(() => bar?.classList.remove('fl-pulse'), 700);
    } else if (flPayload?.fl_stats?.personal_updates) {
      el.textContent = `Personal head: ${flPayload.fl_stats.personal_updates} lần cập nhật`;
    }
  }

  async _onBehavior(productId, action) {
    const product = productId ? this._findProduct(productId) : null;
    const title = product?.title?.slice(0, 48) || (productId ? String(productId).slice(0, 24) : '');
    const entryId = this.activityLog.pushPending(action, productId, title);

    const res = await this.api.trackBehavior(productId, action);
    this.activityLog.complete(
      entryId,
      action,
      productId,
      title,
      res,
      !this.api.isOffline
    );

    if (res?.fl_update) {
      this._updateFlStatus(res.fl_update);
      if (res.fl_update.updated) {
        this.showToast('🧠 AI đã học từ hành vi của bạn (FedPer local)', 'success');
        await this.recWidget.loadRecommendations(productId);
        this.activityLog.logInfo(
          'recommend',
          `Gợi ý làm mới sau FL update · client #${this.auth.session?.client_id ?? '?'}`,
          title
        );
      }
    }
  }

  // ── Products ──
  async loadProducts(category = null) {
    const grid = document.getElementById('productGrid');
    if (!grid) return;

    // Show skeletons
    grid.innerHTML = Array(8).fill(`
      <div class="product-card skeleton">
        <div class="skeleton-image"></div>
        <div class="skeleton-line" style="width:80%"></div>
        <div class="skeleton-line short"></div>
        <div class="skeleton-line price"></div>
      </div>
    `).join('');

    const data = await this.api.fetchProducts(category, 1, this.currentSort);

    if (data && data.products) {
      this.allProducts = data.products.length > 0 ? data.products : MOCK_PRODUCTS;
      if (!this.api.isOffline && data.products.length > 0) {
        this.filteredProducts = data.products;
      } else {
        this.filteredProducts = category
          ? MOCK_PRODUCTS.filter(p => p.category === category)
          : [...MOCK_PRODUCTS];
      }
    }

    this._sortProducts();
    this._renderProducts();
    this._updateConnectionStatus();
  }

  _sortProducts() {
    switch (this.currentSort) {
      case 'price-asc':
        this.filteredProducts.sort((a, b) => a.price - b.price);
        break;
      case 'price-desc':
        this.filteredProducts.sort((a, b) => b.price - a.price);
        break;
      case 'rating':
        this.filteredProducts.sort((a, b) => b.rating - a.rating);
        break;
      case 'name':
        this.filteredProducts.sort((a, b) => a.title.localeCompare(b.title, 'vi'));
        break;
      default:
        // keep original order
        break;
    }
  }

  _renderProducts() {
    const grid = document.getElementById('productGrid');
    if (!grid) return;

    if (this.filteredProducts.length === 0) {
      grid.innerHTML = `
        <div style="grid-column: 1/-1; text-align:center; padding:64px 24px; color:var(--text-muted);">
          <div style="font-size:3rem; margin-bottom:16px;">🔍</div>
          <p>Không tìm thấy sản phẩm nào</p>
        </div>`;
      return;
    }

    grid.innerHTML = this.filteredProducts.map((product, i) => {
      const catInfo = getCategoryInfo(product.category);
      const pid = escapeAttr(product.id);
      return `
        <div class="product-card" data-product-id="${pid}" style="animation-delay:${i * 0.05}s; cursor:pointer">
          <div class="product-card-image">
            ${productImgTag(product)}
            <span class="product-card-badge ${getCategoryBadgeClass(product.category)}" style="background:${catInfo.color}20; color:${catInfo.color}">
              ${catInfo.icon} ${escapeHtml(catInfo.name)}
            </span>
            <button type="button" class="product-card-wishlist" data-action="wishlist">♡</button>
          </div>
          <div class="product-card-body">
            <span class="product-card-category">${escapeHtml(catInfo.name)}</span>
            <h3 class="product-card-title">${escapeHtml(product.title)}</h3>
            <div class="product-card-rating">
              ${generateStars(product.rating)}
              <span class="review-count">(${product.review_count})</span>
            </div>
            <div class="product-card-price">
              <span class="price-current">${formatPrice(product.price)}</span>
            </div>
            <div class="product-card-actions">
              <button type="button" class="btn-add-cart" data-action="cart">🛒 Thêm vào giỏ</button>
              <button type="button" class="btn-quick-view" data-action="view">👁️</button>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  // ── Categories ──
  _renderCategories() {
    const grid = document.getElementById('categoryGrid');
    if (!grid) return;

    grid.innerHTML = CATEGORIES.map(cat => {
      const pool = this.allProducts.length ? this.allProducts : MOCK_PRODUCTS;
      const count = pool.filter(p => p.category === cat.id).length;
      return `
        <div class="category-card" data-category="${escapeAttr(cat.id)}" role="button" tabindex="0"
             style="--card-accent-gradient: linear-gradient(135deg, ${cat.color}08, ${cat.color}15)">
          <span class="category-icon">${cat.icon}</span>
          <span class="category-name">${cat.name}</span>
          <span class="category-count">${count} sản phẩm</span>
        </div>
      `;
    }).join('');
  }

  filterByCategory(categoryId) {
    const locked = this.auth.session?.category;
    if (locked && categoryId !== locked) {
      this.showToast('Demo: chỉ xem sản phẩm trong silo client đã đăng nhập', 'info');
      return;
    }
    if (this.currentCategory === categoryId) {
      this.currentCategory = locked || null;
    } else {
      this.currentCategory = categoryId;
    }

    // Update active state
    document.querySelectorAll('.category-card').forEach(card => {
      card.classList.toggle('active', card.dataset.category === this.currentCategory);
    });

    this.loadProducts(this.currentCategory);
  }

  // ── Product Detail Modal ──
  showProductDetail(productId) {
    const product = this._findProduct(productId);
    if (!product) return;

    this._onBehavior(productId, 'click');
    this.modalViewStart = Date.now();
    this.currentModalProductId = productId;

    const modal = document.getElementById('productModal');
    const body = document.getElementById('modalBody');
    const similarGrid = document.getElementById('similarGrid');
    const catInfo = getCategoryInfo(product.category);

    body.innerHTML = `
      <div class="modal-image">
        ${productImgTag(product)}
      </div>
      <div class="modal-info">
        <span class="modal-category">${catInfo.icon} ${escapeHtml(catInfo.name)}</span>
        <h2 class="modal-title">${escapeHtml(product.title)}</h2>
        <div class="modal-rating">
          ${generateStars(product.rating)}
          <span class="rating-value">${product.rating}</span>
          <span class="review-count">(${product.review_count} đánh giá)</span>
        </div>
        <div class="modal-price">${formatPrice(product.price)}</div>
        <p class="modal-description">${escapeHtml(product.description || 'Sản phẩm trong silo federated client của bạn.')}</p>
        <button type="button" class="modal-add-cart" id="modalAddCartBtn" data-product-id="${escapeAttr(product.id)}">
          🛒 Thêm vào giỏ hàng
        </button>
      </div>
    `;

    const pool = this.allProducts.length ? this.allProducts : MOCK_PRODUCTS;
    const similar = pool
      .filter(p => p.category === product.category && p.id !== product.id)
      .sort(() => Math.random() - 0.5)
      .slice(0, 4);

    similarGrid.innerHTML = similar.map(p => `
      <div class="similar-card" data-product-id="${escapeAttr(p.id)}" role="button" tabindex="0">
        ${productImgTag(p)}
        <div class="similar-card-info">
          <div class="similar-card-title">${escapeHtml(p.title)}</div>
          <div class="similar-card-price">${formatPrice(p.price)}</div>
        </div>
      </div>
    `).join('');

    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Request fresh recommendations based on this product
    this.recWidget.loadRecommendations(productId);
  }

  closeModal() {
    const modal = document.getElementById('productModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';

    if (this.currentModalProductId && this.modalViewStart) {
      this._onBehavior(this.currentModalProductId, 'view');
      this.currentModalProductId = null;
      this.modalViewStart = null;
    }
  }

  // ── Cart ──
  addToCart(productId) {
    const product = this._findProduct(productId);
    if (!product) return;

    this.cart.add(product);
    this.showToast(`Đã thêm "${product.title}" vào giỏ hàng`, 'success');
    this._onBehavior(productId, 'add_to_cart');
  }

  toggleCart() {
    const sidebar = document.getElementById('cartSidebar');
    const overlay = document.getElementById('cartOverlay');
    const isActive = sidebar.classList.contains('active');

    if (isActive) {
      sidebar.classList.remove('active');
      overlay.classList.remove('active');
      document.body.style.overflow = '';
    } else {
      this.cart.renderCart();
      sidebar.classList.add('active');
      overlay.classList.add('active');
      document.body.style.overflow = 'hidden';
    }
  }

  // ── Search ──
  async handleSearch(query) {
    if (!query || query.trim().length === 0) {
      this.currentCategory = null;
      document.querySelectorAll('.category-card').forEach(c => c.classList.remove('active'));
      this.filteredProducts = [...MOCK_PRODUCTS];
      this._sortProducts();
      this._renderProducts();
      return;
    }

    this.activityLog.pushPending('search', null, `"${query.slice(0, 40)}"`);
    const searchRes = await this.api.trackBehavior(null, 'search', null, { query });
    this.activityLog.complete(
      null,
      'search',
      null,
      query.slice(0, 40),
      searchRes,
      !this.api.isOffline
    );
    const data = await this.api.searchProducts(query);

    if (data && data.products) {
      this.filteredProducts = data.products;
    } else {
      const q = query.toLowerCase();
      this.filteredProducts = MOCK_PRODUCTS.filter(p =>
        p.title.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q)
      );
    }

    this._sortProducts();
    this._renderProducts();

    // Scroll to products
    document.getElementById('productsSection')?.scrollIntoView({ behavior: 'smooth' });
  }

  // ── Toast ──
  showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const icons = { success: '✅', error: '❌', info: '💡' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span class="toast-icon">${icons[type] || '💡'}</span>
      <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('removing');
      toast.addEventListener('animationend', () => toast.remove());
    }, 3500);
  }

  // ── Connection Status ──
  _updateConnectionStatus() {
    const banner = document.getElementById('offlineBanner');
    if (banner) {
      banner.classList.toggle('show', this.api.isOffline);
    }
  }

  // ── Find Product ──
  _findProduct(productId) {
    return MOCK_PRODUCTS.find(p => p.id === productId) ||
           this.allProducts.find(p => p.id === productId);
  }

  _bindDelegatedClicks() {
    const handleProductArea = (e) => {
      const actionBtn = e.target.closest('[data-action]');
      const card = e.target.closest('[data-product-id]');
      if (!card?.dataset.productId) return;
      const id = card.dataset.productId;

      if (actionBtn) {
        e.preventDefault();
        e.stopPropagation();
        const act = actionBtn.dataset.action;
        if (act === 'cart') {
          this.addToCart(id);
          return;
        }
        if (act === 'view') {
          this.showProductDetail(id);
          return;
        }
        if (act === 'wishlist') {
          this.showToast('Đã thêm vào yêu thích ❤️', 'info');
          return;
        }
      }
      this.showProductDetail(id);
    };

    ['productGrid', 'recProducts'].forEach((id) => {
      document.getElementById(id)?.addEventListener('click', handleProductArea);
    });

    document.getElementById('similarGrid')?.addEventListener('click', handleProductArea);

    document.getElementById('categoryGrid')?.addEventListener('click', (e) => {
      const card = e.target.closest('.category-card[data-category]');
      if (card?.dataset.category) {
        this.filterByCategory(card.dataset.category);
      }
    });

    document.getElementById('cartItems')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-cart-action]');
      if (!btn?.dataset.productId) return;
      const id = btn.dataset.productId;
      const act = btn.dataset.cartAction;
      if (act === 'minus') {
        this.cart.updateQty(id, -1);
        this.cart.renderCart();
      } else if (act === 'plus') {
        this.cart.updateQty(id, 1);
        this.cart.renderCart();
      } else if (act === 'remove') {
        this.cart.remove(id);
        this.cart.renderCart();
      }
    });

    document.getElementById('modalBody')?.addEventListener('click', (e) => {
      const btn = e.target.closest('#modalAddCartBtn');
      if (btn?.dataset.productId) {
        this.addToCart(btn.dataset.productId);
        this.closeModal();
      }
    });
  }

  // ── Event Bindings ──
  _bindEvents() {
    this._bindDelegatedClicks();

    // Cart button
    document.getElementById('cartBtn')?.addEventListener('click', () => this.toggleCart());
    document.getElementById('cartClose')?.addEventListener('click', () => this.toggleCart());
    document.getElementById('cartOverlay')?.addEventListener('click', () => this.toggleCart());

    // Modal close
    document.getElementById('modalClose')?.addEventListener('click', () => this.closeModal());
    document.getElementById('productModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'productModal') this.closeModal();
    });

    // Search
    const searchBar = document.getElementById('searchBar');
    if (searchBar) {
      const debouncedSearch = debounce((q) => this.handleSearch(q), 400);
      searchBar.addEventListener('input', (e) => debouncedSearch(e.target.value));
      searchBar.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          this.handleSearch(e.target.value);
        }
      });
    }

    // Sort
    document.getElementById('sortSelect')?.addEventListener('change', (e) => {
      this.currentSort = e.target.value;
      this._sortProducts();
      this._renderProducts();
    });

    document.getElementById('btnLogout')?.addEventListener('click', () => {
      this.auth.clear();
      this.recWidget.stopAutoRefresh();
      this.activityLog.hide();
      document.body.classList.remove('has-fl-bar');
      location.reload();
    });

    document.getElementById('btnSwitchAccount')?.addEventListener('click', async () => {
      // Switch without full reload: clear current session and show login overlay again.
      this.auth.clear();
      this.recWidget.stopAutoRefresh();
      this.activityLog.hide();
      document.body.classList.remove('has-fl-bar');
      const bar = document.getElementById('flStatusBar');
      bar?.setAttribute('hidden', '');
      // Bring overlay back (we reuse the existing login UI)
      const overlay = document.getElementById('loginOverlay');
      if (overlay) {
        overlay.style.display = '';
        overlay.classList.add('active');
      }
      await this._showLogin();
    });

    // Checkout
    document.getElementById('checkoutBtn')?.addEventListener('click', () => {
      if (this.cart.getCount() === 0) {
        this.showToast('Giỏ hàng trống! Hãy thêm sản phẩm.', 'error');
        return;
      }
      this.showToast('🎉 Cảm ơn bạn! Đơn hàng đã được ghi nhận (demo).', 'success');
      this.cart.items = [];
      this.cart._save();
      this.cart.renderCart();
      this.toggleCart();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        const modal = document.getElementById('productModal');
        if (modal?.classList.contains('active')) {
          this.closeModal();
        } else {
          const sidebar = document.getElementById('cartSidebar');
          if (sidebar?.classList.contains('active')) {
            this.toggleCart();
          }
        }
      }
    });
  }

  // ── Scroll Animations ──
  _setupScrollAnimations() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    );

    document.querySelectorAll('.section').forEach((el) => {
      el.classList.add('animate-on-scroll', 'visible');
      observer.observe(el);
    });
  }

  // ── Navbar Scroll Effect ──
  _setupNavbarScroll() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;

    let lastScroll = 0;
    window.addEventListener('scroll', () => {
      const currentScroll = window.scrollY;
      navbar.classList.toggle('scrolled', currentScroll > 50);
      lastScroll = currentScroll;
    }, { passive: true });
  }
}

// ─── INITIALIZE APP ─────────────────────────────────
let app;
document.addEventListener('DOMContentLoaded', () => {
  try {
    app = new App();
    window.app = app;
  } catch (err) {
    console.error('FedShop init failed:', err);
    const hint = document.getElementById('loginHint');
    if (hint) {
      hint.textContent = 'Lỗi khởi tạo JS: ' + err.message + ' — thử Ctrl+F5';
    }
  }
});
