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

# Experiment 0
# FIND_BETTER_SCHEMA_PROMPT = '''
#     根据给定的新闻段落、相关的知识图谱三元组，以及SCHEMA，请评估知识图谱三元组是否正确匹配给定新闻段落。正确匹配的定义是：知识图谱三元组可以结构化地表示给定新闻段落的信息，而且保留了大部分新闻段落的内容。如果知识图谱三元组正确匹配给定新闻段落，请回复「任务完成」。如果匹配不正确，请提供一个优化后的SCHEMA，以抽取更好的知识图谱三元组。SCHEMA应采用{'头实体类型': ['关系1', '关系2', ...]}的格式。请注意，如果给定的知识图谱三元组为空([])，你必须调整SCHEMA，且新的SCHEMA不能与之前相同。请尽量提高知识图谱三元组的质量。
#     给定的资料范例如下：
#     <新闻段落>
#     2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。
#     <知识图谱三元组>
#     [(2022语言与智能技术竞赛, 主办方, 中国中文信息学会), (2022语言与智能技术竞赛, 主办方, 中国计算机学会), (2022语言与智能技术竞赛, 已举办次数, 4届), (2022语言与智能技术竞赛, 承办方, 百度公司), (2022语言与智能技术竞赛, 承办方, 中国计算机学会自然语言处理专委会), (2022语言与智能技术竞赛, 承办方, 中国中文信息学会评测工作委员会)]
#     <SCHEMA>
#     {'竞赛名称': ['主办方', '承办方', '已举办次数']}
#     以下为你要分析的资料，请根据以上规则回覆你的答案：
#     <新闻段落>
#     {INPUT1}
#     <知识图谱三元组>
#     {INPUT2}
#     <SCHEMA>
#     {INPUT3}
# '''

# Experiment 1
# FIND_BETTER_SCHEMA_PROMPT = '''
#     给定一段新闻段落、数个与给定之新闻段落相关的知识图谱三元组(头实体, 关系, 尾实体)以及一组SCHEMA，请你评估给定之知识图谱三元组是否为给定新闻段落的最佳结果，最佳结果的定义为：给定之知识图谱三元组可以被视为给定新闻段落的结构化形式，且它有保留大部分给定新闻段落的资讯。若给定之知识图谱三元组是给定新闻段落的最佳结果，请你回覆「任务完成」；反之，若给定之知识图谱三元组不是给定新闻段落的最佳结果，则请你优化给定之SCHEMA，使这个SCHEMA能抽取出更好的知识图谱三元组，并以简体中文回覆你优化过的SCHEMA，SCHEMA请以{'头实体类型': ['关系1', '关系2', ...]}格式表示。请注意：如果给定的知识图谱三元组为空([])，你就一定要调整SCHEMA，回覆的新SCHEMA绝对不能和优化前相同，请尽可能地提升知识图谱三元组的品质。
#     给定的资料范例如下：
#     <新闻段落>
#     2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。
#     <知识图谱三元组>
#     [(2022语言与智能技术竞赛, 主办方, 中国中文信息学会), (2022语言与智能技术竞赛, 主办方, 中国计算机学会), (2022语言与智能技术竞赛, 已举办次数, 4届), (2022语言与智能技术竞赛, 承办方, 百度公司), (2022语言与智能技术竞赛, 承办方, 中国计算机学会自然语言处理专委会), (2022语言与智能技术竞赛, 承办方, 中国中文信息学会评测工作委员会)]
#     <SCHEMA>
#     {'竞赛名称': ['主办方', '承办方', '已举办次数']}
#     以下为你要分析的资料，请根据以上规则回覆你的答案：
#     <新闻段落>
#     {INPUT1}
#     <知识图谱三元组>
#     {INPUT2}
#     <SCHEMA>
#     {INPUT3}
# '''

# Experiment 2
# FIND_BETTER_SCHEMA_PROMPT = '''
#     根据给定的新闻段落、相关的知识图谱三元组，以及SCHEMA，请评估给定的知识图谱三元组是否正确匹配给定新闻段落。正确匹配的定义是：给定的知识图谱三元组可以结构化地表示给定新闻段落的信息，而且保留了大部分新闻段落的内容。评估后你有两种行为可以二择一。第一种行为：如果给定的知识图谱三元组正确匹配给定新闻段落，请只回复「任务完成」。第二种行为：如果匹配不正确，请提供一个优化后的SCHEMA，优化的目标是让下一轮对话使用者给定知识图谱三元组更好（使用者会根据你给予的优化后的SCHEMA训练模型，并产生新的给定知识图谱三元组。）。 SCHEMA应采用{'头实体类型': ['关系1', '关系2', ...]}的格式。请注意，如果给定的知识图谱三元组为空([])，你必须调整SCHEMA，新的SCHEMA不能与给定的SCHEMA相同，且绝对不能回覆「任务完成」。此外，若本次对话你已对SCHEMA进行过修改，就绝对不能再回覆「任务完成」，也就是说，一轮对话的操作流程是先评估知识图谱三元组，再从「回覆任务完成」和「优化SCHEMA」中选一个行为，而不是两个行为都做。
#     给定的资料范例如下：
#     <新闻段落>
#     2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。
#     <知识图谱三元组>
#     [(2022语言与智能技术竞赛, 主办方, 中国中文信息学会), (2022语言与智能技术竞赛, 主办方, 中国计算机学会), (2022语言与智能技术竞赛, 已举办次数, 4届), (2022语言与智能技术竞赛, 承办方, 百度公司), (2022语言与智能技术竞赛, 承办方, 中国计算机学会自然语言处理专委会), (2022语言与智能技术竞赛, 承办方, 中国中文信息学会评测工作委员会)]
#     <SCHEMA>
#     {'竞赛名称': ['主办方', '承办方', '已举办次数']}
#     以下为你要分析的资料，请根据以上规则回覆你的答案：
#     <新闻段落>
#     {INPUT1}
#     <知识图谱三元组>
#     {INPUT2}
#     <SCHEMA>
#     {INPUT3}
# '''

# Experiment 3
# FIND_BETTER_SCHEMA_PROMPT = '''
#     根据使用者给定的新闻段落、知识图谱三元组，以及SCHEMA，请评估给定的知识图谱三元组是否正确匹配给定新闻段落。正确匹配的定义是：给定的知识图谱三元组可以结构化地表示给定新闻段落的信息，而且保留了大部分新闻段落的意義。评估后你有两种行为可以二择一。第一种行为：如果给定的知识图谱三元组正确匹配给定新闻段落，请只回复「任务完成」。第二种行为：如果匹配不正确，请提供一个优化后的SCHEMA，优化的目标是让下一轮对话使用者给定知识图谱三元组更好（使用者本地端的語言模型会根据你提供的优化后的SCHEMA，产生新的知识图谱三元组，再回傳請你評估結果。）。请注意，如果给定的知识图谱三元组为空([])，定义为不正确匹配，若為此特殊情況，你需要有創造力地提供新SCHEMA，請盡力讓使用者本地端的語言模型能通過你的SCHEMA，生成有效的知識圖譜三元組。
#     给定的资料范例如下：
#     <新闻段落>
#     2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。
#     <知识图谱三元组>
#     [(2022语言与智能技术竞赛, 主办方, 中国中文信息学会), (2022语言与智能技术竞赛, 主办方, 中国计算机学会), (2022语言与智能技术竞赛, 已举办次数, 4届), (2022语言与智能技术竞赛, 承办方, 百度公司), (2022语言与智能技术竞赛, 承办方, 中国计算机学会自然语言处理专委会), (2022语言与智能技术竞赛, 承办方, 中国中文信息学会评测工作委员会)]
#     <SCHEMA>
#     {'竞赛名称': ['主办方', '承办方', '已举办次数']}
#     以下为你要分析的资料，请根据以上规则回覆你的答案：
#     <新闻段落>
#     {INPUT1}
#     <知识图谱三元组>
#     {INPUT2}
#     <SCHEMA>
#     {INPUT3}
# '''

# Experiment 4
# FIND_BETTER_SCHEMA_PROMPT = '''
# 你的任务是帮助优化知识图谱三元组，以更好地匹配新闻段落。给定的新闻段落、知识图谱三元组和SCHEMA如下：

# <新闻段落>
# {INPUT1}

# <知识图谱三元组>
# {INPUT2}

# <SCHEMA>
# {INPUT3}

# 请评估给定的知识图谱三元组是否正确匹配给定的新闻段落，正确匹配的定义是：给定的知识图谱三元组可以结构化地「完整」表示给定的新闻段落的信息，并保留了「大部分」给定的新闻段落的意义。请注意，若给定的知识图谱三元组不能代表给定的新闻段落的完整意义，应视为不正确匹配。

# 根据你的评估，你有以下行为可以择一：

# 1. 如果给定的知识图谱三元组正确匹配给定的新闻段落，请回复「任务完成」。

# 2. 如果匹配不正确，请提供一个优化后的SCHEMA，目标是使给定的知识图谱三元组更好地匹配给定的新闻段落。请考虑如何提供更具体、完整和创造性的信息以改进匹配。

# 3. 如果给定的知识图谱三元组为空([])，也定义为不正确匹配，此时你需要破坏性创新，不要基于给定的SCHEMA优化，而是有创造力地基于给定的新闻段落提供新SCHEMA。

# 以下为范例：

# <范例新闻段落>
# 2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。

# <范例知识图谱三元组>
# [(2022语言与智能技术竞赛, 主办方, 中国中文信息学会), (2022语言与智能技术竞赛, 主办方, 中国计算机学会), (2022语言与智能技术竞赛, 已举办次数, 4届), (2022语言与智能技术竞赛, 承办方, 百度公司), (2022语言与智能技术竞赛, 承办方, 中国计算机学会自然语言处理专委会), (2022语言与智能技术竞赛, 承办方, 中国中文信息学会评测工作委员会)]

# <范例SCHEMA>
# {'竞赛名称': ['主办方', '承办方', '已举办次数']}

# 在此范例中，你会先评估给定的知识图谱三元组是否正确匹配给定的新闻段落，若正确匹配，你会回应「任务完成」，若不正确匹配，你会提供一个基于给定的SCHEMA优化的SCHEMA，让使用者后续应用。请注意，如果给定的知识图谱三元组为空 ([])，也定义为不正确匹配，此时您需要有创造力地提供新SCHEMA。
# '''

# Experiment 5
FIND_BETTER_SCHEMA_PROMPT = '''
你的任务是帮助优化知识图谱三元组，以更好地匹配给定的新闻段落。给定的新闻段落、知识图谱三元组和SCHEMA如下：

<新闻段落>
{INPUT1}

<知识图谱三元组>
{INPUT2}

<SCHEMA>
{INPUT3}

请评估 <知识图谱三元组> 是否正确匹配 <新闻段落>，正确匹配的定义是：<知识图谱三元组> 可以结构化地「完整」表示 <新闻段落> 的信息，并保留了「大部分」<新闻段落> 的意义。请注意，若 <知识图谱三元组> 不能代表 <新闻段落> 的完整意义，应视为不正确匹配。

根据你的评估，你有以下行为可以择一：

1. 如果 <知识图谱三元组> 正确匹配 <新闻段落>，请回复「任务完成」。

2. 如果匹配不正确，请提供一个优化后的SCHEMA，目标是使 <知识图谱三元组> 更好地匹配 <新闻段落>。请考虑如何提供更具体、完整和创造性的信息以改进匹配。

3. 如果<知识图谱三元组> 为空([])，也定义为不正确匹配，此时你需要破坏性创新，不要基于<SCHEMA> 优化，而是有创造力地基于<新闻段落>提供新SCHEMA。

以下为范例：

<范例新闻段落>
2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。

<范例知识图谱三元组>
[(2022语言与智能技术竞赛, 主办方, 中国中文信息学会), (2022语言与智能技术竞赛, 主办方, 中国计算机学会), (2022语言与智能技术竞赛, 已举办次数, 4届), (2022语言与智能技术竞赛, 承办方, 百度公司), (2022语言与智能技术竞赛, 承办方, 中国计算机学会自然语言处理专委会), (2022语言与智能技术竞赛, 承办方, 中国中文信息学会评测工作委员会)]

<范例SCHEMA>
{'竞赛名称': ['主办方', '承办方', '已举办次数']}

在此范例中，你会先评估 <范例知识图谱三元组> 是否正确匹配 <范例新闻段落>，若正确匹配，你会回应「任务完成」，若不正确匹配，你会提供一个基于 <范例SCHEMA> 优化的SCHEMA，让使用者后续应用。请注意，如果 <范例知识图谱三元组> 为空 ([])，也定义为不正确匹配，此时您需要有创造力地提供新SCHEMA。
'''
