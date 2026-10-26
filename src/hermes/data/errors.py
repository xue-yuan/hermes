class DataError(RuntimeError):
    """資料相關錯誤的共同父類別。"""


class MissingDataError(DataError):
    """本機沒有需要的資料，要先用 hermes fetch 或 hermes update 下載。"""


class UnknownStockError(DataError):
    """查無此股票代號。"""


class UnsupportedMarketError(DataError):
    """不是上市股票（例如上櫃、興櫃）。hermes 只處理上市股票。"""
