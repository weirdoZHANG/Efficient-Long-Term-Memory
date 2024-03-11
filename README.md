（Efficient Long Term Memory）ELTM公式

![1710125274984](https://github.com/weirdoZHANG/Efficient-Long-Term-Memory/assets/142579062/dfb5c90a-d909-467d-8803-1e2d8aeb4c44)

【代码里是英文】
候选重置输入门->
候选隐藏状态->
重置候选隐藏状态->
单独重置输入x->
单独重置输入ht-1->
更新输出门->
更新单独x的输出门->
更新单独ht-1的输出门->
更新重置候选隐藏状态的输出门->
最终隐藏状态输出->
其中重要的限制条件α + β + γ = 1  
