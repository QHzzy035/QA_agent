# Python 异步编程

## 为什么需要异步

调用一次外部 API 要等两秒，如果程序里有几十个这样的调用，串行执行就要等一分钟。这两秒里 CPU 几乎什么都不做，纯粹在等网络。

异步编程要解决的就是这件事：**等待期间不空转，先去处理别的任务。**

## async / await 的本质

`async def` 定义的函数调用后不会立即执行，而是返回一个协程对象。协程必须交给事件循环才会真正跑起来。

`await` 的作用是「在这里挂起，把控制权还给事件循环，等结果好了再回来」。挂起点就是其他任务得以插队执行的位置。

一个常见的误解是：**写了 `async def` 就自动并发了。** 并不是。下面这段依然是串行的：

```python
results = [await fetch(url) for url in urls]
```

因为每个 `await` 都当场等完了才进入下一次循环。要真正并发，必须把多个协程一起交给事件循环：

```python
results = await asyncio.gather(*(fetch(url) for url in urls))
```

## 阻塞调用会卡死事件循环

事件循环是单线程的。在协程里直接调用阻塞函数（`time.sleep`、同步的 `requests.get`、繁重的 CPU 计算），整个事件循环都会被卡住，所有其他任务一并停摆。

三种处理方式：

- 用异步版本的库（`asyncio.sleep`、`httpx.AsyncClient`）
- 用 `asyncio.to_thread()` 把阻塞调用丢进线程池
- CPU 密集任务用 `run_in_executor` 或干脆换成多进程

这个问题很隐蔽：代码能跑通、结果也对，只是并发完全失效，表现成「不知道为什么还是这么慢」。

## 三个常用工具

**超时**——外部调用必须设超时，否则一个卡住的请求会让任务永远挂着：

```python
await asyncio.wait_for(call(), timeout=30)
```

**限流**——并发太高会触发对方 API 的速率限制：

```python
sem = asyncio.Semaphore(5)
async with sem:
    await call()
```

**分批**——任务特别多时，一次性创建上万协程会吃光内存，可以用 `asyncio.as_completed` 边完成边处理。
