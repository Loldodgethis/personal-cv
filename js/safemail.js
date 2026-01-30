/**
 * SafeMail — AI Phishing Email Detector (client-side)
 * Runs a small local ML model (vocab + coefficients) in the browser; no data sent to any server.
 * Falls back to rule-based analysis if the model file is not loaded.
 */

(function () {
  'use strict';

  var model = null; // { vocab: string[], coef: number[], intercept: number }

  function tokenize(text) {
    return String(text || '').toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(Boolean);
  }

  /** Simple stem so "suspended"/"clicking" match model words "suspend"/"click". */
  function stem(word) {
    if (!word || word.length < 4) return word;
    if (word.length > 5 && (word.slice(-3) === 'ing')) return word.slice(0, -3);
    if (word.length > 4 && (word.slice(-2) === 'ed')) return word.slice(0, -2);
    if (word.length > 5 && (word.slice(-2) === 'ly')) return word.slice(0, -2);
    return word;
  }

  function sigmoid(x) {
    if (x !== x) return 0.5;
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    var e = Math.exp(x);
    return e / (1 + e);
  }

  /** Run the small local model: score = intercept + sum(coef[i] * count[i]), prob = sigmoid(score). */
  function predictWithModel(text) {
    if (!model || !model.vocab || !model.coef) return null;
    var intercept = Number(model.intercept);
    if (intercept !== intercept) return null;
    var words = tokenize(text);
    var counts = model.vocab.map(function () { return 0; });
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      var idx = model.vocab.indexOf(w);
      if (idx !== -1) counts[idx] += 1;
      else {
        var s = stem(w);
        if (s !== w) { idx = model.vocab.indexOf(s); if (idx !== -1) counts[idx] += 1; }
      }
    }
    var score = intercept;
    for (var j = 0; j < model.coef.length; j++) score += Number(model.coef[j]) * counts[j];
    if (score !== score) return null;
    var probPhishing = sigmoid(score);
    var prediction = probPhishing >= 0.5 ? 'Phishing' : 'Safe';
    var confidence = prediction === 'Phishing' ? probPhishing : 1 - probPhishing;
    confidence = Number(confidence);
    if (confidence !== confidence) confidence = 0.5;
    confidence = Math.max(0, Math.min(1, confidence));
    return {
      prediction: prediction,
      confidence: Math.round(confidence * 100) / 100,
      explanation: 'Local ML model (logistic regression) classified this text based on word patterns. ' +
        (prediction === 'Phishing' ? 'Consider verifying the sender and avoiding links or sharing data.' : 'No strong phishing signals; still verify important requests through official channels.')
    };
  }

  /** Rule-based scoring using taxonomy: urgency, threats, authority, credentials, CTA, financial, attachment, TLDs, composite. */
  function analyzeWithRules(text) {
    var lower = text.toLowerCase();
    var score = 0;
    var reasons = [];

    // 1. Urgency & time pressure
    var URGENCY = /\b(urgent|immediately|action\s*required|final\s*notice|last\s*warning|respond\s*now|within\s*(24|12)\s*hours|time-?sensitive|expires\s*today|deadline|now\s*or\s*never|failure\s*to\s*comply|limited\s*time|pending\s*suspension|overdue|must\s*be\s*reviewed\s*today|avoid\s*penalties)\b/gi;
    if (URGENCY.test(lower)) { score += 20; reasons.push('urgency or time pressure'); }

    // 2. Fear, threats & consequences
    var THREATS = /\b(account\s*(suspended|locked|closure)|unauthorized\s*access|suspicious\s*activity|security\s*breach|compromised|unusual\s*activity|violation\s*detected|policy\s*violation|permanent\s*closure|loss\s*of\s*(access|funds)|legal\s*action|report\s*to\s*authorities|restricted\s*access|frozen\s*funds|penalties)\b/gi;
    if (THREATS.test(lower)) { score += 25; reasons.push('threats or consequences'); }

    // 3. Authority / impersonation
    var AUTHORITY = /\b(security\s*(team|department|operations)|IT\s*department|support\s*team|administrator|compliance\s*office|billing\s*department|accounting\s*team|payroll|HR\s*department|secure\s*(billing|server)|trusted\s*partner)\b/gi;
    if (AUTHORITY.test(lower)) { score += 15; reasons.push('authority or impersonation language'); }

    // 4. Credential harvesting
    var CREDENTIALS = /\b(verify\s*(your\s*)?(identity|account|email)|confirm\s*(your\s*)?(account|password|email|payment)|re-?authenticate|validate\s*credentials|login\s*required|sign\s*in\s*to\s*continue|update\s*account\s*information|reset\s*required|security\s*verification|authentication\s*failure|password|credentials|social\s*security|ssn|credit\s*card|cvv|email\s*and\s*password)\b/gi;
    if (CREDENTIALS.test(lower)) { score += 22; reasons.push('credential or verification request'); }

    // 5. Call-to-action (high weight when unsolicited)
    var CTA = /\b(click\s*(here|the\s*link|below)|tap\s*here|open\s*(the\s*)?attachment|download\s*now|verify\s*now|confirm\s*now|update\s*now|review\s*immediately|access\s*document|restore\s*access|secure\s*your\s*account|follow\s*the\s*instructions)\b/gi;
    if (CTA.test(lower)) { score += 22; reasons.push('call-to-action (click/open/verify)'); }

    // 6. Financial lures (required/overdue/pending = risk; "statement available" / "payroll processed" = often legit)
    var FINANCIAL = /\b(invoice\s*(attached|is\s*overdue)|payment\s*(required|overdue|pending)|billing\s*issue|refund\s*pending|charge\s*detected|transaction\s*failed|wire\s*transfer|direct\s*deposit\s*update|tax\s*refund|payroll\s*update|gift\s*cards|cryptocurrency)\b/gi;
    if (FINANCIAL.test(lower)) { score += 20; reasons.push('financial or payment language'); }

    // 7. Attachment & file language (malware phishing)
    var ATTACHMENT = /\b(attached\s*(document|invoice|file)|see\s*(the\s*)?attachment|open\s*(the\s*)?(attachment|file)|secure\s*document|encrypted\s*attachment|enable\s*macros|enable\s*editing|compressed\s*file|password-?protected\s*file)\b/gi;
    if (ATTACHMENT.test(lower)) { score += 18; reasons.push('attachment or file instruction'); }

    // 8. Suspicious file extensions (very high weight — .exe, .scr, etc.)
    var BAD_EXT = /\.(exe|scr|js|iso|img|bat|cmd|vbs|wsf)(\s|$|[^\w])|\.(zip|rar)\s|invoice[_\s]*\d+\.exe|\.pdf\.exe|double\s*extension/i;
    if (BAD_EXT.test(lower) || /\.exe\b/i.test(text)) { score += 35; reasons.push('suspicious file extension (e.g. .exe)'); }

    // 9. Link & URL / TLDs
    var SUSPICIOUS_TLD = /\.(ru|tk|ml|ga|cf|gq|xyz|top|cn)\b|paypal-alerts|secure-login-verify|company-billing-support|billing-support|account\s*closure|loss\s*of\s*funds/i;
    if (SUSPICIOUS_TLD.test(text)) { score += 28; reasons.push('suspicious domain or TLD'); }
    var IP_IN_URL = /https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/;
    if (IP_IN_URL.test(text)) { score += 25; reasons.push('link to an IP address'); }
    var SHORTENED = /\b(bit\.ly|tinyurl|t\.co|goo\.gl)\b/i;
    if (SHORTENED.test(text)) { score += 12; reasons.push('shortened URL'); }
    var LINK_PATTERN = /https?:\/\/[^\s<>"']+/gi;
    var links = text.match(LINK_PATTERN) || [];
    if (links.length > 3) { score += 10; reasons.push('multiple links'); }

    // 10. Composite high-risk phrases (gold signals)
    var COMPOSITE = /verify\s*(your\s*)?account\s*immediately|unusual\s*activity\s*detected|failure\s*to\s*respond\s*will\s*result|click\s*below\s*to\s*restore|login\s*required\s*to\s*avoid\s*suspension|attached\s*invoice\s*(is\s*)?overdue|confirm\s*your\s*identity\s*to\s*continue|payment\s*is\s*overdue|open\s*the\s*attachment\s*and\s*follow/gi;
    if (COMPOSITE.test(lower)) { score += 25; reasons.push('high-risk phrase combination'); }

    // 11. Reward / prize (existing)
    var REWARD = /\b(free\s*(money|prize|gift)|winner|congratulations|you\s*(have\s*)?been\s*selected|claim\s*(your\s*)?reward)\b/gi;
    if (REWARD.test(lower)) { score += 18; reasons.push('prize or reward wording'); }

    // —— Legit discriminators (reduce false positives: SAFE-1..SAFE-7 anchor — low urgency, optional navigation, no credential verbs)
    var legitDeduction = 0;
    var legitReasons = [];
    // Strong anchor: "no action required" or optional navigation only (SAFE-1..SAFE-7) — large deduction so we don't overfit to keywords
    var STRONG_LEGIT = /no further action (is )?required|no action (is )?(needed|required)|informational purposes only|no action items required|there are no action items|details are available (on|at)|you can review account activity (anytime)?|you can access (this file|your (account )?settings)/gi;
    if (STRONG_LEGIT.test(lower)) { legitDeduction += 28; legitReasons.push('no action required / informational'); }
    // Informational tone: optional action, no threat
    var LEGIT_PHRASES = /if this was you|if this wasn't you|you may ignore|ignore this message|no changes required|(information is )?already (changed|correct)|you can retry|at your convenience|visit your (bank )?app or website|visit your website|unless your information is outdated|you can review (account )?activity (anytime)?|sign in through your bank's official website/gi;
    if (LEGIT_PHRASES.test(lower)) { legitDeduction += 14; legitReasons.push('informational tone, optional action'); }
    // Canonical / trusted domains (link points to known-good host)
    var CANONICAL_DOMAIN = /account\.microsoft\.com|account\.google\.com|login\.microsoftonline\.com|microsoft\.com\/|google\.com|drive\.google\.com|docusign\.net|chase\.com|\.edu(\/|\s|$)|workday\.|it\.university\.edu/i;
    if (CANONICAL_DOMAIN.test(text)) { legitDeduction += 18; legitReasons.push('canonical or trusted domain in link'); }
    score = Math.max(0, score - legitDeduction);

    score = Math.min(100, score);
    var confidence = score / 100;
    var prediction = score >= 52 ? 'Phishing' : 'Safe';
    if (prediction === 'Safe') confidence = 1 - confidence;
    var explanation;
    if (reasons.length === 0 && legitReasons.length === 0) {
      explanation = 'No strong phishing patterns were found.';
    } else if (prediction === 'Safe' && legitReasons.length > 0) {
      explanation = (reasons.length > 0 ? 'Phishing-style signals were present; ' : '') +
        'legit discriminators (e.g. informational tone, trusted domain) reduced the score. ' +
        (reasons.length > 0 ? 'Signals: ' + reasons.join('; ') + '. ' : '') +
        'Legit: ' + legitReasons.join('; ') + '. Still verify links and sender when in doubt.';
    } else {
      explanation = 'Signals detected: ' + reasons.join('; ') + '. ' +
        (prediction === 'Phishing' ? 'Consider verifying the sender and avoiding links or attachments.' : 'Still verify important requests through official channels.');
    }
    return { prediction: prediction, confidence: Math.round(confidence * 100) / 100, explanation: explanation };
  }

  /** Gold phish signals: if present, we don't override to Safe even when legit framing exists. */
  function hasGoldPhishSignals(text) {
    var lower = (text || '').toLowerCase();
    if (/\.exe\b/i.test(text)) return true;
    if (/verify\s*(your\s*)?(identity|account)\s*immediately|failure\s*to\s*respond\s*will\s*result|click\s*below\s*to\s*restore|login\s*required\s*to\s*avoid|attached\s*invoice\s*(is\s*)?overdue|open\s*the\s*attachment\s*and\s*follow|confirm\s*your\s*(password|identity)/gi.test(lower)) return true;
    return false;
  }

  /** Legit discriminators: override Phishing → Safe when strong legit framing and no gold phish. SAFE-1..SAFE-7 anchor. */
  function hasLegitDiscriminators(text) {
    var lower = (text || '').toLowerCase();
    var strongLegit = /no further action (is )?required|no action (is )?(needed|required)|informational purposes only|no action items required|there are no action items|you can review account activity (anytime)?|you can access (this file|your (account )?settings)|details are available/gi.test(lower);
    var legitPhrase = /if this was you|if this wasn't you|you may ignore|no changes required|already (changed|correct)|you can retry|at your convenience|visit your (bank )?app or website|sign in through your bank's official website/gi.test(lower);
    var canonicalDomain = /account\.microsoft\.com|account\.google\.com|login\.microsoftonline\.com|microsoft\.com\/|google\.com|drive\.google\.com|docusign\.net|chase\.com|\.edu(\/|\s|$)|workday\.|it\.university\.edu/i.test(text);
    return !hasGoldPhishSignals(text) && (strongLegit || (legitPhrase && canonicalDomain));
  }

  function analyzeEmail(emailText) {
    var text = String(emailText || '').trim();
    if (!text) {
      return { prediction: 'Safe', confidence: 0, explanation: 'No text provided. Paste email content to analyze.' };
    }
    var out = predictWithModel(text);
    if (!out) out = analyzeWithRules(text);
    // Override: if model/rules said Phishing but strong legit framing (no action required / informational) and no gold phish, mark Safe
    if (out.prediction === 'Phishing' && hasLegitDiscriminators(text)) {
      out = {
        prediction: 'Safe',
        confidence: 0.68,
        explanation: 'Phishing-style keywords were present, but strong legit framing (e.g. "no action required", informational tone) and no high-risk signals indicate a likely safe message. Still verify links and sender when in doubt.'
      };
    }
    return out;
  }

  function runCheck() {
    var input = document.getElementById('safemail-input');
    var resultEl = document.getElementById('safemail-result');
    var badgeEl = document.getElementById('safemail-badge');
    var confEl = document.getElementById('safemail-confidence');
    var explEl = document.getElementById('safemail-explanation');

    var text = input && input.value ? input.value.trim() : '';
    if (!text) {
      resultEl.hidden = false;
      resultEl.removeAttribute('data-risk');
      badgeEl.textContent = 'No input';
      confEl.textContent = '';
      explEl.textContent = 'Paste email content above, then click Check Email.';
      return;
    }

    var out = analyzeEmail(text);
    var conf = out.confidence;
    if (conf !== conf || conf < 0 || conf > 1) conf = 0.5;
    resultEl.hidden = false;
    resultEl.setAttribute('data-risk', out.prediction.toLowerCase());
    badgeEl.textContent = 'Result: ' + out.prediction;
    confEl.textContent = 'Confidence: ' + Math.round(conf * 100) + '%';
    explEl.textContent = out.explanation;
  }

  function toggleInfo() {
    var btn = document.getElementById('safemail-info-toggle');
    var content = document.getElementById('safemail-info-content');
    if (!btn || !content) return;
    var open = content.hidden;
    content.hidden = !open;
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  function loadModel() {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', 'js/safemail-model.json', true);
    xhr.onload = function () {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          model = JSON.parse(xhr.responseText);
        } catch (e) { /* keep null, use rules */ }
      }
    };
    xhr.onerror = function () { /* use rules */ };
    xhr.send();
  }

  function init() {
    loadModel();
    var checkBtn = document.getElementById('safemail-check');
    var infoBtn = document.getElementById('safemail-info-toggle');
    if (checkBtn) checkBtn.addEventListener('click', runCheck);
    if (infoBtn) infoBtn.addEventListener('click', toggleInfo);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
