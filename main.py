# ============================================================================
# BitCraft Market Analyzer - Complete Desktop Application
# ============================================================================

# ============================================================================
# config.py
# ============================================================================

API_BASE_URL = "https://bitjita.com"
DEFAULT_MIN_VOLUME = 100
DEFAULT_MAX_RESULTS = 100
REQUEST_TIMEOUT = 10

# Outlier filtering settings - more aggressive
OUTLIER_PERCENTILE_LOW = 10   # Remove bottom 10% of prices
OUTLIER_PERCENTILE_HIGH = 75  # Remove top 25% of prices (extreme listings)

# BitCraft has 9 regions (R0-R8)
BITCRAFT_REGIONS = [
    "All Regions",
    "Region 0 (R0)",
    "Region 1 (R1)", 
    "Region 2 (R2)",
    "Region 3 (R3)",
    "Region 4 (R4)",
    "Region 5 (R5)",
    "Region 6 (R6)",
    "Region 7 (R7)",
    "Region 8 (R8)"
]

# ============================================================================
# utils/logger.py
# ============================================================================

import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """Set up application logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

# ============================================================================
# core/models.py
# ============================================================================

from dataclasses import dataclass

@dataclass
class MarketItem:
    """Market item data model"""
    item_id: str
    name: str
    price: float
    volume: int
    region: str
    order_type: str  # 'sell' or 'buy'

# ============================================================================
# api/bitjita_client.py
# ============================================================================

import requests
from typing import List, Dict, Optional
import time

logger = setup_logger('bitjita_client')

class BitjitaClient:
    """Client for interacting with Bitjita API"""
    
    def __init__(self, base_url: str, timeout: int = REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BitCraft-Market-Analyzer',
            'x-app-identifier': 'BitCraft-Market-Analyzer'
        })
        self._item_cache = None
        self._market_data_cache = {}  # Cache for processed market data
        self._cache_timestamp = {}
    
    def get_regions(self) -> List[str]:
        """Return BitCraft regions (R0-R8)"""
        return BITCRAFT_REGIONS
    
    def get_all_items(self) -> List[Dict]:
        """Get list of all items with names for search"""
        if self._item_cache:
            return self._item_cache
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/market",
                params={'hasOrders': 'true'},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            items = []
            if isinstance(data, dict):
                items = data.get('data', {}).get('items', [])
            elif isinstance(data, list):
                items = data
            
            self._item_cache = items
            return items
        except Exception as e:
            logger.error(f"Failed to fetch items: {e}")
            return []
    
    def get_cached_market_data(self, region_name: str) -> Optional[List[Dict]]:
        """Get cached market data if available and recent (within 5 minutes)"""
        import time
        cache_key = region_name
        
        if cache_key in self._market_data_cache:
            cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)
            if cache_age < 300:  # 5 minutes
                logger.info(f"Using cached data for {region_name} (age: {int(cache_age)}s)")
                return self._market_data_cache[cache_key]
        
        return None
    
    def cache_market_data(self, region_name: str, data: List[Dict]):
        """Cache market data with timestamp"""
        import time
        cache_key = region_name
        self._market_data_cache[cache_key] = data
        self._cache_timestamp[cache_key] = time.time()
        logger.info(f"Cached {len(data)} orders for {region_name}")
    
    def get_market_data(self, region_name: str, item_name: Optional[str] = None, use_cache: bool = True, progress_callback=None) -> List[Dict]:
        """
        Fetch market listings for a region, optionally filtered by item name
        Uses cache if available and recent
        progress_callback: optional function to report progress (item_num, total_items)
        """
        # Check cache first
        if use_cache and not item_name:  # Only use cache for full region scans
            cached = self.get_cached_market_data(region_name)
            if cached:
                if item_name and item_name != "All Items":
                    # Filter cached data by item name
                    return [order for order in cached if order.get('name') == item_name]
                return cached
        
        all_orders = []
        
        try:
            # Extract region number
            region_num = None
            if region_name != "All Regions" and "(R" in region_name and ")" in region_name:
                start = region_name.index("(R") + 2
                end = region_name.index(")", start)
                region_num = region_name[start:end].strip()
            
            logger.info(f"Fetching market data for {region_name} (region {region_num})...")
            
            # Get items
            params = {'hasOrders': 'true'}
            if region_num:
                params['regionId'] = region_num
            
            response = self.session.get(
                f"{self.base_url}/api/market",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            items = []
            if isinstance(data, dict):
                items = data.get('data', {}).get('items', [])
            elif isinstance(data, list):
                items = data
            
            logger.info(f"Received {len(items)} items with orders")
            
            # Filter by item name if specified
            if item_name and item_name != "All Items":
                items = [item for item in items if item.get('name') == item_name]
                logger.info(f"Filtered to {len(items)} items matching '{item_name}'")
            
            # Filter items that have orders
            items_with_orders = [
                item for item in items 
                if item.get('hasSellOrders') or item.get('hasBuyOrders')
            ]
            
            # Process items
            # Process ALL items but limit display later based on max_results
            items_to_process = items_with_orders  # Process all items
            
            logger.info(f"Processing {len(items_to_process)} items...")
            
            for idx, item in enumerate(items_to_process):
                item_id = str(item.get('id', ''))
                item_name_api = item.get('name', 'Unknown Item')
                
                # Report progress
                if progress_callback:
                    progress_callback(idx + 1, len(items_to_process))
                
                try:
                    detail_params = {}
                    if region_num:
                        detail_params['regionId'] = region_num
                    
                    detail_response = self.session.get(
                        f"{self.base_url}/api/market/item/{item_id}",
                        params=detail_params,
                        timeout=self.timeout
                    )
                    detail_response.raise_for_status()
                    detail_data = detail_response.json()
                    
                    # Extract orders - check both structures
                    sell_orders = []
                    buy_orders = []
                    
                    if 'data' in detail_data:
                        sell_orders = detail_data.get('data', {}).get('sellOrders', [])
                        buy_orders = detail_data.get('data', {}).get('buyOrders', [])
                    else:
                        sell_orders = detail_data.get('sellOrders', [])
                        buy_orders = detail_data.get('buyOrders', [])
                    
                    # Process sell orders
                    for order in sell_orders:
                        order_region = str(order.get('regionId', ''))
                        if region_num and order_region != region_num:
                            continue
                        
                        # API uses 'priceThreshold' and quantity is a string
                        price = float(order.get('priceThreshold', 0))
                        quantity = int(order.get('quantity', 0))
                        
                        if price > 0 and quantity > 0:
                            order_detail = {
                                'claim_name': order.get('claimName', 'Unknown'),
                                'owner': order.get('ownerUsername', 'Unknown'),
                                'price': price,
                                'quantity': quantity,
                                'region_name': order.get('regionName', region_name)
                            }
                            all_orders.append({
                                'item_id': item_id,
                                'name': item_name_api,
                                'price': price,
                                'volume': quantity,
                                'region': order.get('regionName', region_name),
                                'order_type': 'sell',
                                'claim_name': order.get('claimName', 'Unknown'),
                                'owner': order.get('ownerUsername', 'Unknown')
                            })
                    
                    # Process buy orders
                    for order in buy_orders:
                        order_region = str(order.get('regionId', ''))
                        if region_num and order_region != region_num:
                            continue
                        
                        price = float(order.get('priceThreshold', 0))
                        quantity = int(order.get('quantity', 0))
                        
                        if price > 0 and quantity > 0:
                            all_orders.append({
                                'item_id': item_id,
                                'name': item_name_api,
                                'price': price,
                                'volume': quantity,
                                'region': order.get('regionName', region_name),
                                'order_type': 'buy',
                                'claim_name': order.get('claimName', 'Unknown'),
                                'owner': order.get('ownerUsername', 'Unknown')
                            })
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(items_to_process)} items, found {len(all_orders)} orders")
                    
                    time.sleep(0.01)  # Reduced to 0.01 for even faster processing
                    
                except Exception as e:
                    logger.debug(f"Error processing item {item_id}: {e}")
                    continue
            
            logger.info(f"Processed {len(all_orders)} market entries from {region_name}")
            
            # Cache the results if this was a full scan (no item filter)
            if not item_name or item_name == "All Items":
                self.cache_market_data(region_name, all_orders)
            
            return all_orders
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return []

# ============================================================================
# core/processor.py
# ============================================================================

from typing import List, Dict
from collections import defaultdict
import statistics

logger_proc = setup_logger('processor')

class DataProcessor:
    """Processes and aggregates market data"""
    
    @staticmethod
    def remove_outliers(prices: List[float]) -> List[float]:
        """
        Remove price outliers using percentile filtering
        More aggressive filtering to remove unrealistic high prices
        """
        if len(prices) < 3:
            return prices
        
        # Sort prices
        sorted_prices = sorted(prices)
        
        # Calculate percentile indices
        low_idx = int(len(sorted_prices) * (OUTLIER_PERCENTILE_LOW / 100))
        high_idx = int(len(sorted_prices) * (OUTLIER_PERCENTILE_HIGH / 100))
        
        # Ensure we have at least some data
        if high_idx <= low_idx:
            high_idx = low_idx + 1
        
        # Return middle range
        filtered = sorted_prices[low_idx:high_idx]
        
        if not filtered:
            return prices[:1] if prices else prices
        
        logger_proc.debug(f"Filtered {len(prices)} prices to {len(filtered)} (removed {len(prices) - len(filtered)} outliers)")
        return filtered
    
    @staticmethod
    def aggregate_items(items: List[MarketItem], order_filter: str = 'both') -> Dict:
        """
        Group by item_id and calculate metrics with outlier removal
        order_filter: 'both', 'sell', or 'buy'
        Also stores detailed order information per claim
        """
        aggregated = defaultdict(lambda: {
            'name': '',
            'sell_prices': [],
            'sell_volumes': [],
            'buy_prices': [],
            'buy_volumes': [],
            'sell_details': [],  # Store detailed sell orders
            'buy_details': []    # Store detailed buy orders
        })
        
        for item in items:
            agg = aggregated[item.item_id]
            agg['name'] = item.name
            
            if item.order_type == 'sell':
                agg['sell_prices'].append(item.price)
                agg['sell_volumes'].append(item.volume)
                # Store details: we'll get this from raw data
            elif item.order_type == 'buy':
                agg['buy_prices'].append(item.price)
                agg['buy_volumes'].append(item.volume)
        
        result = {}
        for item_id, data in aggregated.items():
            # Filter based on order type
            prices_to_use = []
            volumes_to_use = []
            
            if order_filter == 'sell':
                prices_to_use = data['sell_prices']
                volumes_to_use = data['sell_volumes']
            elif order_filter == 'buy':
                prices_to_use = data['buy_prices']
                volumes_to_use = data['buy_volumes']
            else:  # 'both'
                prices_to_use = data['sell_prices'] + data['buy_prices']
                volumes_to_use = data['sell_volumes'] + data['buy_volumes']
            
            if not prices_to_use:
                continue
            
            # Remove price outliers for more accurate averages
            filtered_prices = DataProcessor.remove_outliers(prices_to_use)
            
            total_volume = sum(volumes_to_use)
            avg_price = sum(filtered_prices) / len(filtered_prices) if filtered_prices else 0
            value_score = total_volume * avg_price
            
            result[item_id] = {
                'name': data['name'],
                'total_volume': total_volume,
                'avg_price': avg_price,
                'value_score': value_score,
                'sell_orders': len(data['sell_prices']),
                'buy_orders': len(data['buy_prices']),
                'sell_details': data['sell_details'],
                'buy_details': data['buy_details']
            }
        
        logger_proc.info(f"Aggregated {len(result)} unique items (filter: {order_filter})")
        return result

# ============================================================================
# core/analyzer.py
# ============================================================================

from typing import List, Dict

logger_analyze = setup_logger('analyzer')

class MarketAnalyzer:
    """Analyzes and ranks market data"""
    
    @staticmethod
    def find_best_deals(aggregated: Dict, min_volume: int = DEFAULT_MIN_VOLUME) -> List[Dict]:
        """Ranks and filters items by value score"""
        filtered = []
        
        for item_id, data in aggregated.items():
            if data['total_volume'] >= min_volume:
                filtered.append({
                    'item_id': item_id,
                    'name': data['name'],
                    'total_volume': data['total_volume'],
                    'avg_price': data['avg_price'],
                    'value_score': data['value_score'],
                    'sell_orders': data.get('sell_orders', 0),
                    'buy_orders': data.get('buy_orders', 0)
                })
        
        filtered.sort(key=lambda x: x['value_score'], reverse=True)
        
        logger_analyze.info(f"Found {len(filtered)} items meeting criteria (min_volume={min_volume})")
        if filtered:
            logger_analyze.info(f"Top item: {filtered[0]}")
        
        return filtered

# ============================================================================
# gui/worker.py
# ============================================================================

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

class WorkerSignals(QObject):
    """Signals for background worker"""
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(str)
    progress_percent = Signal(int)  # New: for progress bar

class MarketDataWorker(QRunnable):
    """Background worker for fetching and processing market data"""
    
    def __init__(self, client, region: str, min_volume: int, max_results: int, order_filter: str, item_name: str = None):
        super().__init__()
        self.client = client
        self.region = region
        self.min_volume = min_volume
        self.max_results = max_results
        self.order_filter = order_filter
        self.item_name = item_name
        self.signals = WorkerSignals()
    
    @Slot()
    def run(self):
        """Execute the market data fetch and analysis"""
        try:
            self.signals.progress.emit(f"Fetching data for {self.region}...")
            self.signals.progress_percent.emit(0)
            
            # Progress callback for API fetching
            def update_fetch_progress(current, total):
                if total > 0:
                    # Map to 10-80% range for API fetching
                    pct = int(10 + (current / total * 70))
                    self.signals.progress_percent.emit(pct)
                    if current % 50 == 0:  # Update message every 50 items
                        self.signals.progress.emit(f"Processing item {current}/{total}...")
            
            raw_data = self.client.get_market_data(
                self.region, 
                self.item_name,
                progress_callback=update_fetch_progress
            )
            
            if not raw_data:
                self.signals.progress_percent.emit(0)
                self.signals.error.emit("No data returned from API")
                return
            
            self.signals.progress.emit(f"Aggregating {len(raw_data)} orders...")
            self.signals.progress_percent.emit(85)
            
            items = []
            raw_orders_by_item = defaultdict(list)  # Store raw orders grouped by item
            
            for entry in raw_data:
                try:
                    # Handle both cached format and API format
                    item = MarketItem(
                        item_id=str(entry.get('item_id', 'unknown')),
                        name=entry.get('name', 'Unknown Item'),
                        price=float(entry.get('price', 0)),
                        volume=int(entry.get('volume', 0)),
                        region=entry.get('region', self.region),
                        order_type=entry.get('order_type', 'sell')
                    )
                    if item.price > 0 and item.volume > 0:
                        items.append(item)
                        # Store raw order data for details
                        raw_orders_by_item[item.item_id].append(entry)
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Skipping invalid entry: {e}")
                    continue
            
            if not items:
                self.signals.progress_percent.emit(0)
                self.signals.error.emit("No valid market data found")
                return
            
            self.signals.progress.emit(f"Analyzing {len(items)} valid orders...")
            self.signals.progress_percent.emit(90)
            
            processor = DataProcessor()
            aggregated = processor.aggregate_items(items, self.order_filter)
            
            if not aggregated:
                self.signals.progress_percent.emit(0)
                self.signals.error.emit("No items after aggregation")
                return
            
            self.signals.progress.emit(f"Ranking {len(aggregated)} items...")
            self.signals.progress_percent.emit(95)
            
            analyzer = MarketAnalyzer()
            all_results = analyzer.find_best_deals(aggregated, self.min_volume)
            
            # Add detailed order information to each result
            for result in all_results:
                item_id = result['item_id']
                if item_id in raw_orders_by_item:
                    result['order_details'] = raw_orders_by_item[item_id]
                else:
                    result['order_details'] = []
            
            if not all_results:
                self.signals.progress_percent.emit(0)
                self.signals.error.emit(f"No items meet minimum volume of {self.min_volume}")
                return
            
            # Limit display to max results (but we analyzed everything)
            results = all_results[:self.max_results]
            
            logger.info(f"Worker finished - emitting {len(results)} results")
            logger.info(f"About to emit finished signal...")
            
            self.signals.progress_percent.emit(100)
            self.signals.progress.emit(f"Showing top {len(results)} of {len(all_results)} items")
            self.signals.finished.emit(results)
            
            logger.info("Finished signal emitted successfully")
            
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            self.signals.progress_percent.emit(0)
            self.signals.error.emit(str(e))

# ============================================================================
# gui/table_model.py
# ============================================================================

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from typing import List, Dict, Any

class MarketTableModel(QAbstractTableModel):
    """Table model for displaying market analysis results with expandable details"""
    
    HEADERS = ['Item Name', 'Total Volume', 'Average Price', 'Value Score', 'Sell Orders', 'Buy Orders']
    
    def __init__(self):
        super().__init__()
        self._data: List[Dict] = []
        self._expanded_rows = set()  # Track which rows are expanded
    
    def rowCount(self, parent=QModelIndex()) -> int:
        count = 0
        for idx, item in enumerate(self._data):
            count += 1  # Main row
            if idx in self._expanded_rows:
                # Add detail rows
                count += len(item.get('order_details', []))
        return count
    
    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)
    
    def _get_actual_item_and_detail(self, row: int):
        """Map display row to actual item index and detail index"""
        current_row = 0
        for item_idx, item in enumerate(self._data):
            if current_row == row:
                return item_idx, -1, item  # Main row
            current_row += 1
            
            if item_idx in self._expanded_rows:
                detail_count = len(item.get('order_details', []))
                if current_row + detail_count > row:
                    detail_idx = row - current_row
                    return item_idx, detail_idx, item
                current_row += detail_count
        
        return None, None, None
    
    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        
        try:
            item_idx, detail_idx, item = self._get_actual_item_and_detail(row)
            
            if item_idx is None:
                return None
            
            if role == Qt.DisplayRole:
                # Main item row
                if detail_idx == -1:
                    if col == 0:
                        prefix = "▼ " if item_idx in self._expanded_rows else "▶ "
                        return prefix + item.get('name', 'Unknown')
                    elif col == 1:
                        return f"{item.get('total_volume', 0):,}"
                    elif col == 2:
                        return f"${item.get('avg_price', 0):.2f}"
                    elif col == 3:
                        return f"${item.get('value_score', 0):,.2f}"
                    elif col == 4:
                        return str(item.get('sell_orders', 0))
                    elif col == 5:
                        return str(item.get('buy_orders', 0))
                # Detail row
                else:
                    details = item.get('order_details', [])
                    if detail_idx < len(details):
                        order = details[detail_idx]
                        if col == 0:
                            claim = order.get('claim_name', 'Unknown')
                            owner = order.get('owner', 'Unknown')
                            order_type = order.get('order_type', 'sell').upper()
                            return f"    [{order_type}] {claim} ({owner})"
                        elif col == 1:
                            return f"{order.get('volume', 0):,}"
                        elif col == 2:
                            return f"${order.get('price', 0):.2f}"
                        elif col == 3:
                            vol = order.get('volume', 0)
                            price = order.get('price', 0)
                            return f"${vol * price:,.2f}"
                        elif col == 4 or col == 5:
                            return ""
            
            elif role == Qt.FontRole:
                if detail_idx >= 0:
                    # Detail rows use smaller font
                    from PySide6.QtGui import QFont
                    font = QFont()
                    font.setPointSize(9)
                    return font
                    
        except Exception as e:
            logger = setup_logger('table_model')
            logger.error(f"Error rendering cell [{row},{col}]: {e}", exc_info=True)
            return "Error"
        
        return None
    
    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None
    
    def update_data(self, new_data: List[Dict]):
        """Update table with new data"""
        try:
            logger = setup_logger('table_model')
            logger.info(f"update_data called with {len(new_data)} items")
            
            self.beginResetModel()
            self._data = new_data if new_data else []
            self._expanded_rows.clear()  # Clear expansions on new data
            self.endResetModel()
            
            logger.info(f"Table reset complete. rowCount={self.rowCount()}, data items={len(self._data)}")
        except Exception as e:
            logger = setup_logger('table_model')
            logger.error(f"Error updating table data: {e}", exc_info=True)
    
    def toggle_expand(self, row: int):
        """Toggle expansion of a row"""
        item_idx, detail_idx, item = self._get_actual_item_and_detail(row)
        
        # Only toggle if clicking on main row
        if item_idx is not None and detail_idx == -1:
            if item_idx in self._expanded_rows:
                self._expanded_rows.remove(item_idx)
            else:
                self._expanded_rows.add(item_idx)
            
            # Refresh the view
            self.beginResetModel()
            self.endResetModel()
    
    def sort(self, column: int, order: Qt.SortOrder):
        """Sort table by column"""
        self.layoutAboutToBeChanged.emit()
        
        key_map = {
            0: 'name',
            1: 'total_volume',
            2: 'avg_price',
            3: 'value_score',
            4: 'sell_orders',
            5: 'buy_orders'
        }
        
        key = key_map.get(column, 'value_score')
        reverse = (order == Qt.DescendingOrder)
        
        self._data.sort(key=lambda x: x.get(key, 0), reverse=reverse)
        
        self.layoutChanged.emit()

# ============================================================================
# gui/controller.py
# ============================================================================

from PySide6.QtCore import QThreadPool

logger_ctrl = setup_logger('controller')

class MarketController:
    """Controller handling GUI events and coordinating operations"""
    
    def __init__(self, client, table_model, status_callback):
        self.client = client
        self.table_model = table_model
        self.status_callback = status_callback
        self.thread_pool = QThreadPool()
    
    def refresh_market_data(self, region: str, min_volume: int, max_results: int, order_filter: str, item_name: str = None):
        """Trigger background market data refresh"""
        logger_ctrl.info(f"Refreshing market data for {region}, min_volume={min_volume}, max_results={max_results}, order_filter={order_filter}, item={item_name}")
        
        # Clear existing data immediately
        self.table_model.update_data([])
        self.status_callback("Loading...")
        
        worker = MarketDataWorker(self.client, region, min_volume, max_results, order_filter, item_name)
        
        logger_ctrl.info("Connecting worker signals...")
        worker.signals.finished.connect(self._on_data_ready)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(self.status_callback)
        worker.signals.progress_percent.connect(self._on_progress_update)
        
        logger_ctrl.info("Starting worker thread...")
        self.thread_pool.start(worker)
        logger_ctrl.info("Worker thread started")
    
    def _on_progress_update(self, percent: int):
        """Handle progress updates"""
        # This will be connected by MainWindow
        pass
    
    def _on_data_ready(self, results: List[Dict]):
        """Handle completed data fetch"""
        try:
            logger_ctrl.info(f"Controller received {len(results)} results")
            if results:
                logger_ctrl.info(f"First result keys: {list(results[0].keys())}")
                logger_ctrl.info(f"First result has {len(results[0].get('order_details', []))} order details")
            else:
                logger_ctrl.warning("Results list is empty!")
            
            logger_ctrl.info(f"Calling table_model.update_data with {len(results)} items")
            self.table_model.update_data(results)
            logger_ctrl.info(f"Table now has {self.table_model.rowCount()} rows")
            self.status_callback(f"Last updated • Showing {len(results)} items")
        except Exception as e:
            logger_ctrl.error(f"Error updating table: {e}", exc_info=True)
            self.status_callback(f"Error displaying results: {e}")
    
    def _on_error(self, error_msg: str):
        """Handle errors"""
        logger_ctrl.error(f"Error: {error_msg}")
        self.status_callback(f"Error: {error_msg}")

# ============================================================================
# gui/main_window.py
# ============================================================================

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSpinBox, QTableView,
    QLabel, QStatusBar, QCompleter, QProgressBar
)
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, controller, table_model, regions: List[str], client):
        super().__init__()
        self.controller = controller
        self.table_model = table_model
        self.client = client
        
        self.setWindowTitle("BitCraft Market Analyzer")
        self.setMinimumSize(1000, 600)
        
        self._setup_ui(regions)
    
    def _setup_ui(self, regions: List[str]):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Top controls - Row 1
        controls_layout1 = QHBoxLayout()
        
        controls_layout1.addWidget(QLabel("Region:"))
        self.region_combo = QComboBox()
        self.region_combo.addItems(regions)
        controls_layout1.addWidget(self.region_combo)
        
        controls_layout1.addWidget(QLabel("Min Volume:"))
        self.min_volume_spin = QSpinBox()
        self.min_volume_spin.setMinimum(0)
        self.min_volume_spin.setMaximum(100000)
        self.min_volume_spin.setValue(DEFAULT_MIN_VOLUME)
        self.min_volume_spin.setSingleStep(10)
        controls_layout1.addWidget(self.min_volume_spin)
        
        controls_layout1.addWidget(QLabel("Max Results:"))
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setMinimum(10)
        self.max_results_spin.setMaximum(2000)
        self.max_results_spin.setValue(DEFAULT_MAX_RESULTS)
        self.max_results_spin.setSingleStep(50)
        controls_layout1.addWidget(self.max_results_spin)
        
        controls_layout1.addWidget(QLabel("Order Type:"))
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Both", "Sell Orders", "Buy Orders"])
        controls_layout1.addWidget(self.order_type_combo)
        
        controls_layout1.addStretch()
        layout.addLayout(controls_layout1)
        
        # Top controls - Row 2: Item search
        controls_layout2 = QHBoxLayout()
        
        controls_layout2.addWidget(QLabel("Item:"))
        self.item_combo = QComboBox()
        self.item_combo.setEditable(True)
        self.item_combo.setInsertPolicy(QComboBox.NoInsert)
        self.item_combo.addItem("All Items")
        self.item_combo.setMinimumWidth(300)
        
        # Add completer for autocomplete
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.item_combo.setCompleter(completer)
        
        controls_layout2.addWidget(self.item_combo)
        
        self.refresh_btn = QPushButton("Refresh Market Data")
        self.refresh_btn.clicked.connect(self._on_refresh)
        controls_layout2.addWidget(self.refresh_btn)
        
        self.clear_cache_btn = QPushButton("Clear Cache & Rescan")
        self.clear_cache_btn.clicked.connect(self._on_clear_cache)
        controls_layout2.addWidget(self.clear_cache_btn)
        
        controls_layout2.addStretch()
        layout.addLayout(controls_layout2)
        
        # Table view
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setSortingEnabled(False)  # Disable sorting with expandable rows
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.clicked.connect(self._on_table_click)
        
        # Set column widths
        self.table_view.setColumnWidth(0, 250)
        self.table_view.setColumnWidth(1, 120)
        self.table_view.setColumnWidth(2, 120)
        self.table_view.setColumnWidth(3, 150)
        self.table_view.setColumnWidth(4, 100)
        self.table_view.setColumnWidth(5, 100)
        
        layout.addWidget(self.table_view)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Load items after UI is set up
        self._load_items()
    
    def _load_items(self):
        """Load item names for the dropdown"""
        items = self.client.get_all_items()
        item_names = sorted([item.get('name', '') for item in items if item.get('name')])
        
        for name in item_names:
            self.item_combo.addItem(name)
        
        # Update completer
        self.item_combo.completer().setModel(self.item_combo.model())
        
        self.status_bar.showMessage(f"Loaded {len(item_names)} items")
    
    def _on_refresh(self):
        """Handle refresh button click"""
        region = self.region_combo.currentText()
        min_volume = self.min_volume_spin.value()
        max_results = self.max_results_spin.value()
        item = self.item_combo.currentText()
        
        # Map order type selection to filter string
        order_type_map = {
            "Both": "both",
            "Sell Orders": "sell",
            "Buy Orders": "buy"
        }
        order_filter = order_type_map.get(self.order_type_combo.currentText(), "both")
        
        if not item or item == "All Items":
            item = None
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.controller.refresh_market_data(region, min_volume, max_results, order_filter, item)
    
    def _on_clear_cache(self):
        """Clear cache and force fresh API scan"""
        self.client._market_data_cache.clear()
        self.client._cache_timestamp.clear()
        self.status_bar.showMessage("Cache cleared - next refresh will rescan API")
        self._on_refresh()
    
    def update_progress(self, percent: int):
        """Update progress bar"""
        self.progress_bar.setValue(percent)
        if percent >= 100:
            # Hide progress bar after a short delay
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: self.progress_bar.setVisible(False))
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_bar.showMessage(message)
    
    def _on_table_click(self, index):
        """Handle table row clicks for expansion"""
        if index.isValid():
            self.table_model.toggle_expand(index.row())

# ============================================================================
# main.py
# ============================================================================

import sys
from PySide6.QtWidgets import QApplication

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    client = BitjitaClient(API_BASE_URL)
    regions = client.get_regions()
    
    table_model = MarketTableModel()
    window = MainWindow(None, table_model, regions, client)
    
    controller = MarketController(client, table_model, window.update_status)
    
    # Connect progress updates to window
    controller._on_progress_update = window.update_progress
    
    window.controller = controller
    
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()