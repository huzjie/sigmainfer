# @sigmainfer/sdk

SigmaInfer TypeScript 客户端。

```ts
import { SigmaClient } from "@sigmainfer/sdk";
const client = new SigmaClient("http://127.0.0.1:8000");
const res = await client.generate({ prompt: "hello" });
console.log(res.output);
```
