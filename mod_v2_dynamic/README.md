**增加信息维度**： 增加输入信息，ai可以获知对手的剩余手牌数量，对方手牌越少，对于ai的惩罚会乘一个放大倍数，ai手牌越少，奖励会乘一个放大倍数，这个临界值是 5 

**调整出牌策略**： 使用惩罚防止ai过早出比较大的牌，如果有对子或者三代一，但是ai拆开单出，扣一分