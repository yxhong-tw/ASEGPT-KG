TRIPLET_LABELING_PROMPT = '''
    給定一段新聞段落，請幫我從中找出所有的知識圖譜三元組 (頭實體, 關係, 尾實體)。請幫我過濾掉對於構成新聞段落不重要的三元組，並只給我過濾後的結果。 注意：新聞段落內可能有一個以上的三元組存在，若有多個三元組，格式請以[(頭實體1, 關係1, 尾實體1), (頭實體2, 關係2, 尾實體2)]以此類推呈現。
    <新聞段落>
    全球貿易衰減正在重塑全球經濟景況，美國和印度等依靠內需的國家成為贏家，中國和德國等仰賴出口的國家則淪為主要輸家。華爾街日報（WSJ）報導，世界貿易組織（WTO）數據顯示，全球貨物貿易今年第1季比去年第4季減少，延續去年開始的低潮，經濟學家預估，今年都將維持頹勢。全球貿易流動減弱，正在拉大20國集團（G20）之間的成長差距，失去優勢的一方為素來享有貿易順差的出口導向經濟體，成長正落後於美國和印度這類市場龐大、相對仰賴內需的國家。
    <答案>
    [(全球貿易, 衰減, 重塑全球經濟景況), (美國, 依賴, 內需), (印度, 依賴, 內需), (中國, 仰賴, 出口), (德國, 仰賴, 出口), (世界貿易組織, 數據顯示, 全球貨物貿易減少), (全球貿易流動, 減弱, 拉大G20成長差距)]
    <新聞段落>
    {INPUT}
    <答案>
'''

TRIPLET_FULL_LABELING_PROMPT = '''
    給定一段新聞段落，請幫我從中找出所有的知識圖譜三元組 (頭實體, 關係, 尾實體)，如果有你覺得有缺失的關係，請把它補上。 注意：新聞段落內可能有一個以上的三元組存在，若有多個三元組，格式請以[(頭實體1, 關係1, 尾實體1), (頭實體2, 關係2, 尾實體2)]以此類推呈現。
    <新聞段落>
    全球貿易衰減正在重塑全球經濟景況，美國和印度等依靠內需的國家成為贏家，中國和德國等仰賴出口的國家則淪為主要輸家。華爾街日報（WSJ）報導，世界貿易組織（WTO）數據顯示，全球貨物貿易今年第1季比去年第4季減少，延續去年開始的低潮，經濟學家預估，今年都將維持頹勢。全球貿易流動減弱，正在拉大20國集團（G20）之間的成長差距，失去優勢的一方為素來享有貿易順差的出口導向經濟體，成長正落後於美國和印度這類市場龐大、相對仰賴內需的國家。
    <答案>
    [(全球貿易, 衰減, 重塑全球經濟景況), (美國, 依賴, 內需), (印度, 依賴, 內需), (中國, 仰賴, 出口), (德國, 仰賴, 出口), (世界貿易組織, 數據顯示, 全球貨物貿易減少), (全球貿易流動, 減弱, 拉大G20成長差距)]
    <新聞段落>
    {INPUT}
    <答案>
'''