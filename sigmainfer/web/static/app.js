const run = document.getElementById("run");
const output = document.getElementById("output");
const stats = document.getElementById("stats");

run.addEventListener("click", async () => {
  output.textContent = "生成中…";
  stats.textContent = "";
  const body = {
    prompt: document.getElementById("prompt").value,
    block_size: +document.getElementById("block_size").value,
    num_steps: +document.getElementById("num_steps").value,
    max_new_tokens: +document.getElementById("max_new_tokens").value,
  };
  try {
    const res = await fetch("/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    output.textContent = data.output;
    stats.textContent = JSON.stringify(data.stats, null, 2);
  } catch (e) {
    output.textContent = "请求失败: " + e;
  }
});
