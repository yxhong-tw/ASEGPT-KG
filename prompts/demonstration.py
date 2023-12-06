RELATION_EXTRACTION_ONE_SHOT_EXAMPLE = '''
<新聞段落>
全球雲端龍頭亞馬遜 (AMZN-US) 旗下 AWS 今 (21) 日舉辦「製造轉型戰略峯會」，香港暨臺灣區總經理王定愷會後受訪表示，目前 AI 需求還是滿強勁，公司也持續投入自研晶片，看好比泛用型晶片效能更好，更能貼近與滿足各行業需求。\n針對微軟 (MSFT-US) 近期調降輝達 H100 訂單，王定愷指出，不清楚同業動向，但 AWS 也是輝達 (NVDA-US) 與超微 (AMD-US) 的大客戶，看起來 AI 需求還是滿強勁的，並沒有鬆動疲軟跡象。\n研調數據顯示，2022 至 2027 年全球公有雲市場規模將從 4000 億美元一舉提升到 1.2 兆美元，年複合成長率 (CAGR) 高達 25%，長期成長趨勢相當明確。\n此外，王定愷看好，由於 AWS 掌握眾多客戶，瞭解客戶需求，因此除了採購輝達等泛用型晶片外，也攜手臺灣半導體供應鏈生產自研晶片，主要用於 AI 領域的推論 (Inference) 與訓練 (Training)。\n至於市場關注生成式 AI，王定愷認為，整體架構大致分為三層，第一層主要是晶片、負責算力，公司有泛用型與自研晶片的方案，上雲平臺後則是第二層，主要介接分隔個別客戶資料，第三層則是大語言模型，可以提供客戶多元選擇。
<答案>
\"{ \"triplets\":[ \"全球雲端龍頭亞馬遜, 舉辦, 製造轉型戰略峯會\", \"亞馬遜, 旗下, AWS\", \"AWS, 舉辦, 製造轉型戰略峯會\", \"王定愷, 受訪, 目前AI需求滿強勁\", \"AWS, 持續投入, 自研晶片\", \"王定愷, 指出, 不清楚同業動向\", \"AWS, 輝達, 大客戶\", \"AWS, 超微, 大客戶\", \"AI, 需求, 強勁\", \"研調數據, 顯示, 2022至2027年全球公有雲市場規模提升到1.2兆美元\", \"年複合成長率, 高達, 25%\", \"王定愷, 看好, AWS掌握眾多客戶\", \"AWS, 採購, 輝達泛用型晶片\", \"AWS, 攜手, 臺灣半導體供應鏈生產自研晶片用於AI領域推論與訓練\", \"王定愷, 認為, 生成式AI整體架構分為三層\", \"第一層, 主要是, 晶片\", \"第一層, 為, 公司有泛用型與自研晶片的方案\", \"上雲平臺後, 是, 第二層\", \"第二層, 主要介接, 分隔個別客戶資料\", \"第三層, 是, 大語言模型\", \"提供, 客戶, 多元選擇\" ] }\"
'''

RELATION_EXTRACTION_WITH_EXPLANATION_JSON_SCHEMA_ONE_SHOT_EXAMPLE = '''
<新聞段落>
全球雲端龍頭亞馬遜 (AMZN-US) 旗下 AWS 今 (21) 日舉辦「製造轉型戰略峯會」，香港暨臺灣區總經理王定愷會後受訪表示，目前 AI 需求還是滿強勁，公司也持續投入自研晶片，看好比泛用型晶片效能更好，更能貼近與滿足各行業需求。\n針對微軟 (MSFT-US) 近期調降輝達 H100 訂單，王定愷指出，不清楚同業動向，但 AWS 也是輝達 (NVDA-US) 與超微 (AMD-US) 的大客戶，看起來 AI 需求還是滿強勁的，並沒有鬆動疲軟跡象。\n研調數據顯示，2022 至 2027 年全球公有雲市場規模將從 4000 億美元一舉提升到 1.2 兆美元，年複合成長率 (CAGR) 高達 25%，長期成長趨勢相當明確。\n此外，王定愷看好，由於 AWS 掌握眾多客戶，瞭解客戶需求，因此除了採購輝達等泛用型晶片外，也攜手臺灣半導體供應鏈生產自研晶片，主要用於 AI 領域的推論 (Inference) 與訓練 (Training)。\n至於市場關注生成式 AI，王定愷認為，整體架構大致分為三層，第一層主要是晶片、負責算力，公司有泛用型與自研晶片的方案，上雲平臺後則是第二層，主要介接分隔個別客戶資料，第三層則是大語言模型，可以提供客戶多元選擇。
<答案>
\"{ \"triplets\":[ \"全球雲端龍頭亞馬遜, 舉辦, 製造轉型戰略峯會\", \"亞馬遜, 旗下, AWS\", \"AWS, 舉辦, 製造轉型戰略峯會\", \"王定愷, 受訪, 目前AI需求滿強勁\", \"AWS, 持續投入, 自研晶片\", \"王定愷, 指出, 不清楚同業動向\", \"AWS, 輝達, 大客戶\", \"AWS, 超微, 大客戶\", \"AI, 需求, 強勁\", \"研調數據, 顯示, 2022至2027年全球公有雲市場規模提升到1.2兆美元\", \"年複合成長率, 高達, 25%\", \"王定愷, 看好, AWS掌握眾多客戶\", \"AWS, 採購, 輝達泛用型晶片\", \"AWS, 攜手, 臺灣半導體供應鏈生產自研晶片用於AI領域推論與訓練\", \"王定愷, 認為, 生成式AI整體架構分為三層\", \"第一層, 主要是, 晶片\", \"第一層, 為, 公司有泛用型與自研晶片的方案\", \"上雲平臺後, 是, 第二層\", \"第二層, 主要介接, 分隔個別客戶資料\", \"第三層, 是, 大語言模型\", \"提供, 客戶, 多元選擇\" ], "rationals":[ \"全球雲端龍頭亞馬遜 (AMZN-US) 旗下 AWS 今 (21) 日舉辦「製造轉型戰略峯會」\", \"亞馬遜旗下的AWS\", \"AWS舉辦了製造轉型戰略峯會\", \"王定愷表示目前AI需求很強勁\", \"公司持續投入自研晶片\", \"王定愷指出不清楚同業動向\", \"AWS是輝達的大客戶\", \"AWS是超微的大客戶\", \"AI需求還是很強勁的\", \"研調數據顯示2022至2027年全球公有雲市場規模將提升到1.2兆美元，年複合成長率高達25%。\", \"年複合成長率高達25%\", \"王定愷看好AWS掌握眾多客戶\", \"AWS採購輝達等泛用型晶片\", \"AWS攜手臺灣半導體供應鏈生產自研晶片，主要用於AI領域的推論與訓練。\", \"王定愷認為整體架構大致分為三層\", \"第一層主要是晶片\", \"第一層指公司有泛用型與自研晶片的方案\", \"上雲平臺後則是第二層\", \"第二層主要介接分隔個別客戶資料\", \"第三層則是大語言模型\", \"第三層指可以提供客戶多元選擇\" ] }\"
'''
