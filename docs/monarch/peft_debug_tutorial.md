debug 入口：peft/tests/test_initialization.py
对`def test_lora_pissa_linear_init_default` 运行debug
test script中的网络结构如下：
```python
self.linear = nn.Linear(1000, 1000)
                self.embed = nn.Embedding(1000, 1000)
                self.conv2d = nn.Conv2d(100, 100, 3)


