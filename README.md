（Efficient Long Term Memory）ELTM公式

<img width="158" alt="image" src="https://github.com/weirdoZHANG/Efficient-Long-Term-Memory-/assets/142579062/9fc7a401-e40e-4ac2-a654-390ed8e8528e">

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

站是一个例子，使用六大时间序列数据集之一：ILI
![image](https://github.com/weirdoZHANG/Efficient-Long-Term-Memory-/assets/142579062/d76b81f6-8d23-4155-a2e0-a7acffbfc7c0)

![image](https://github.com/weirdoZHANG/Efficient-Long-Term-Memory-/assets/142579062/b110e9a8-e440-4fbd-94db-fa823cc754db)

![image](https://github.com/weirdoZHANG/Efficient-Long-Term-Memory-/assets/142579062/9f88fb32-0471-41ea-b4f5-f912568cd621)




