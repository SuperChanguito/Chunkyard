// Browser regression checks for the QA fixes that live in the page.
//
// 1. Start the fake-embedder server:   uv run python tests/ui/preview_server.py 8766
// 2. Install once (in tests/ui):       npm install
// 3. Run:                              node qa_ui_check.js [http://127.0.0.1:8766] [path-to-chrome-or-edge]
const puppeteer = require("puppeteer-core");

const BASE = process.argv[2] || "http://127.0.0.1:8766";
const BROWSER = process.argv[3] || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
let failed = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failed++; };
const q = (s) => encodeURIComponent(s);

async function selectIn(page, k, rank, phrase) {
  return page.evaluate((k, rank, phrase) => {
    const el = document.querySelector(`.res[data-k="${k}"][data-rank="${rank}"] .text`);
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const i = n.textContent.indexOf(phrase);
      if (i >= 0) {
        const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + phrase.length);
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
        return true;
      }
    }
    return false;
  }, k, rank, phrase);
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: BROWSER, headless: true, args: ["--disable-gpu"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  // Fix 6: elapsed seconds while chunking and embedding.
  const statusTexts = [];
  await page.exposeFunction("recordStatus", (t) => statusTexts.push(t));
  await page.evaluateOnNewDocument(() => {
    new MutationObserver(() => {
      const el = document.getElementById("docInfo");
      if (el && el.className === "busy") window.recordStatus(el.textContent);
    }).observe(document, { subtree: true, childList: true, characterData: true });
  });

  // Fix 1: verdict wording when a #1 is only a heading.
  await page.goto(`${BASE}/?sample=tent&q=${q("What's not covered")}`);
  await page.waitForSelector(".res", { timeout: 60000 });
  ok(statusTexts.some((t) => /\d+s$/.test(t)), "loading status shows elapsed seconds");
  const verdict = await page.$eval("#summary .verdict", (e) => e.textContent);
  ok(!verdict.includes("All three agree"), "heading-only #1 is not reported as agreement");
  ok(verdict.includes("only retrieved the heading “What's not covered”"), "verdict says Paragraph only retrieved the heading");

  // Fix 1: hover-linking follows the server's `matches`.
  const linkOk = await page.evaluate(() => {
    for (const el of document.querySelectorAll(".res")) {
      el.dispatchEvent(new MouseEvent("mouseenter"));
      const r = res.results[el.dataset.k].find((x) => x.rank === +el.dataset.rank);
      const want = new Set(r.matches.map((m) => `${m.strategy}:${m.rank}`));
      const got = new Set([...document.querySelectorAll(".res.linked")].map((o) => `${o.dataset.k}:${o.dataset.rank}`));
      el.dispatchEvent(new MouseEvent("mouseleave"));
      if (want.size !== got.size || [...want].some((x) => !got.has(x))) return false;
    }
    return true;
  });
  ok(linkOk, "hover-linking outlines exactly the server's matching cards");

  // Fix 7: denominator and source-character labels.
  const counts = await page.evaluate(() => [document.querySelector(".pairs").textContent,
    Object.values(res.results).reduce((n, rs) => n + rs.length, 0)]);
  ok(counts[0].includes(`of ${counts[1]} results found by only one strategy`), `"only one strategy" count uses ${counts[1]} real results`);
  ok(await page.$$eval(".reshead .meta", (els) => els.every((e) => e.textContent.endsWith("source chars"))), "cards label lengths as source chars");
  ok(await page.$$eval(".stat .sub[title]", (els) => els.length === 3 && els.every((e) => e.textContent.includes("source chars"))), "stats label lengths as source chars");

  // Fix 2: answers must be selected words, not whole chunks.
  await page.goto(`${BASE}/?sample=tent&q=${q("How long is the warranty?")}`);
  await page.waitForSelector(".res .mark", { timeout: 60000 });
  await page.evaluate(() => window.getSelection().removeAllRanges());
  await page.click('.res[data-k="section"][data-rank="1"] .mark');
  ok(await page.$eval("#askErr", (e) => !e.classList.contains("hidden") && e.textContent.includes("select the words")),
    "clicking without a selection asks for a selection");
  ok((await page.$$(".colverdict")).length === 0, "nothing is saved without a selection");
  const long = await page.evaluate(() => {
    const el = document.querySelector('.res[data-k="fixed"][data-rank="1"] .text');
    const r = document.createRange(); r.selectNodeContents(el);
    const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
    return s.toString().trim().length;
  });
  await page.click('.res[data-k="fixed"][data-rank="1"] .mark');
  ok(long > 250 && await page.$eval("#askErr", (e) => e.textContent.includes("at most 250 characters")),
    `a ${long}-character selection is refused (limit 250 = half the fixed size)`);
  const phrase = "two years from the date of purchase";
  const where = await page.evaluate((phrase) => {
    for (const el of document.querySelectorAll(".res .text")) {
      if (el.textContent.includes(phrase)) { const c = el.closest(".res"); return [c.dataset.k, +c.dataset.rank]; }
    }
    return null;
  }, phrase);
  if (where && await selectIn(page, where[0], where[1], phrase)) {
    await page.click(`.res[data-k="${where[0]}"][data-rank="${where[1]}"] .mark`);
    await page.waitForSelector(".colverdict", { timeout: 30000 });
    ok((await page.$eval("#answerBar", (e) => e.textContent)).includes(phrase), "a short selection is saved as the answer");
  } else {
    ok(false, "couldn't find the warranty phrase in the results to select");
  }

  // Fix 5: a slow earlier response never overwrites a later question.
  await page.setRequestInterception(true);
  let first = true;
  page.on("request", (r) => {
    if (r.url().endsWith("/api/query") && first) { first = false; setTimeout(() => r.continue(), 2500); }
    else r.continue();
  });
  const chips = await page.$$(".chip");
  await chips[1].click();                       // slow request
  await new Promise((r) => setTimeout(r, 150));
  const disabledWhileBusy = await page.$$eval(".chip, .qlink", (els) => els.every((e) => e.disabled));
  await page.evaluate(() => { document.querySelectorAll(".chip").forEach((c) => (c.disabled = false)); });
  await chips[2].click();                       // fast request
  await new Promise((r) => setTimeout(r, 3500));
  const shown = await page.evaluate(() => [res.question, document.getElementById("question").value]);
  ok(disabledWhileBusy, "example chips and saved-question links are disabled while a request is in flight");
  ok(shown[0] === shown[1], `results match the question in the box ("${shown[1]}")`);

  ok(errors.length === 0, "no JavaScript errors" + (errors.length ? ": " + errors.join(" | ") : ""));
  await browser.close();
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error("ERROR", e); process.exit(1); });
