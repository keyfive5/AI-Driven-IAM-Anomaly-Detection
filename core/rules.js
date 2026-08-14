/**
 * Detection rule pack — the deterministic half of the system.
 *
 * Unsupervised models find *unusual*. Rules encode *known-bad*, which is a
 * different question and the reason pure-ML detection stacks disappoint in
 * production: `cloudtrail:StopLogging` is not statistically interesting in an
 * estate where nobody has ever done it — it is simply an emergency.
 *
 * Every rule
 *   - reads only past-and-present state (the same streaming context the
 *     features use), so nothing here could be run only in hindsight;
 *   - carries an ATT&CK tactic and technique, so alerts sequence into a chain;
 *   - asserts a *risk floor* rather than an absolute score, letting the models
 *     push a finding higher but never silently bury it.
 */

const rx = (s) => new RegExp(s, 'i');

/** Count how many times an action appears in the identity's recent window. */
function windowCount(ctx, pattern) {
  let c = 0;
  for (const a of ctx.windowActions) if (pattern.test(a)) c++;
  return c;
}

/**
 * Is there anything odd about the circumstances, independent of the operation?
 *
 * This predicate is why the rule pack does not drown a SOC. An identity
 * administrator calling `iam:CreateUser` at 11:00 from the office, on a session
 * with MFA, doing what they did last Tuesday, is *their job*. The same call at
 * 03:00 from a new address on a session with no second factor is an incident.
 * Broad-surface rules therefore fire on the circumstances, never on the verb
 * alone; only operations that are wrong under every circumstance
 * (`cloudtrail:StopLogging`, sharing a snapshot outside the account) fire
 * unconditionally.
 */
function contextStrength(e, ctx) {
  let s = 0;
  if (ctx.newAction) s += 2;
  if (ctx.newIp) s += 1.5;
  if (ctx.newCountry) s += 2.5;
  if (ctx.travelKmh > 400) s += 3;
  if (ctx.hour < 6 || ctx.hour >= 21) s += 1;
  if (ctx.mfaDropOff) s += 1.5;
  if (e.outcome === 'failure') s += 1;
  if (ctx.dormantRevival) s += 2;
  if (ctx.burst5m >= 10) s += 1.5;
  if (ctx.failStreak >= 3) s += 1.5;
  return s;
}

const suspiciousContext = (e, ctx) => contextStrength(e, ctx) > 0;

/**
 * Has this identity made this exact call often enough for it to be their job?
 *
 * Without this test the pack re-alerts on every onboarding an identity
 * administrator performs — the single most common reason a cloud detection
 * rule gets switched off in production.
 */
const isRoutineFor = (ctx, min = 5) => ctx.actionCountForActor >= min;

/**
 * Scale a contextual rule's risk floor by how much is actually odd.
 * One weak signal (a change at 21:00) must not score the same as four
 * (new country, no MFA, dormant identity, denied attempt) — otherwise the
 * queue sorts by rule identity instead of by how alarming the finding is.
 */
function gradedRisk(base, strength, floorFraction = 0.78) {
  const scale = floorFraction + (1 - floorFraction) * Math.min(1, strength / 3);
  return Math.round(base * scale);
}

/** Which of those circumstances actually applied, for the alert text. */
function contextReason(e, ctx) {
  if (ctx.newAction) return 'the first time this identity has ever made this call';
  if (ctx.travelKmh > 400) return 'a location the identity could not physically have reached';
  if (ctx.newCountry) return `a country this identity has never worked from (${e.country || 'unknown'})`;
  if (ctx.newIp) return `an address this identity has never used (${e.ip})`;
  if (ctx.hour < 6 || ctx.hour >= 21) return `${String(ctx.hour).padStart(2, '0')}:00 UTC, outside working hours`;
  if (ctx.mfaDropOff) return 'a session with no second factor, unlike this identity\'s norm';
  if (e.outcome === 'failure') return 'an attempt that was denied';
  if (ctx.dormantRevival) return 'an identity that had gone quiet for over a week';
  if (ctx.burst5m >= 10) return 'the middle of an unusual burst of activity';
  return 'unusual circumstances';
}

function rawText(e) {
  if (!e.raw) return '';
  try { return JSON.stringify(e.raw); } catch { return ''; }
}

/**
 * risk: the floor this rule asserts, 0–100.
 * severity: 1 (informational) … 5 (page someone).
 */
export const RULES = [
  {
    id: 'logging_tamper',
    name: 'Audit logging disabled or deleted',
    tactic: 'Defense Evasion',
    technique: 'T1562.008',
    severity: 5,
    risk: 96,
    when: (e) => /^(cloudtrail:(StopLogging|DeleteTrail|PutEventSelectors|UpdateTrail)|guardduty:(DeleteDetector|UpdateDetector)|config:Delete|logs:DeleteLogGroup)/i.test(e.action)
      && { detail: `${e.action} against ${e.resource} — the estate stops recording what happens next.` },
  },
  {
    id: 'mfa_removed',
    name: 'MFA device removed',
    tactic: 'Defense Evasion',
    technique: 'T1556.006',
    severity: 5,
    risk: 92,
    when: (e) => /(DeactivateMFADevice|DeleteVirtualMFADevice|DeleteAuthenticationMethod|ResetMFADevice)/i.test(e.action)
      && { detail: `MFA removed from ${e.resource}. Standard first move after a session is stolen.` },
  },
  {
    id: 'admin_policy_attached',
    name: 'Administrator policy attached',
    tactic: 'Privilege Escalation',
    technique: 'T1098.003',
    severity: 5,
    risk: 90,
    when: (e) => (/^iam:(Attach(User|Role|Group)Policy|Put(User|Role|Group)Policy|SetDefaultPolicyVersion)/i.test(e.action)
      || /(GrantPrivilege|roleAssignments\/write|SetIamPolicy)/i.test(e.action))
      && (rx('admin|poweruser|\\*|FullAccess|Owner|superuser').test(e.resource) || rx('"Action"\\s*:\\s*"\\*"').test(rawText(e)))
      && { detail: `${e.actor} granted ${e.resource}. Full-control grants are the shortest path from foothold to takeover.` },
  },
  {
    id: 'privilege_grant',
    name: 'Identity permissions modified under unusual circumstances',
    tactic: 'Privilege Escalation',
    technique: 'T1098',
    severity: 4,
    risk: 74,
    when: (e, ctx) => (/^iam:(Attach|Put|Add|Create|Update)(User|Role|Group|Policy|LoginProfile|AssumeRolePolicy)/i.test(e.action)
      || /^okta:(GrantPrivilege|AddGroupMembership|CreateUser|CreateApiToken|ResetPassword)/i.test(e.action)
      || /(roleAssignments\/write|SetIamPolicy|roleDefinitions\/write)/i.test(e.action))
      && suspiciousContext(e, ctx)
      && (contextStrength(e, ctx) >= 2 || !isRoutineFor(ctx, 5))
      && {
        detail: `${e.action} on ${e.resource}, from ${contextReason(e, ctx)}.`,
        risk: gradedRisk(74, contextStrength(e, ctx)),
      },
  },
  {
    id: 'access_key_minted',
    name: 'Long-lived access key created',
    tactic: 'Persistence',
    technique: 'T1098.001',
    severity: 4,
    risk: 80,
    when: (e, ctx) => /^iam:CreateAccessKey/i.test(e.action)
      && (suspiciousContext(e, ctx) || ctx.priorSensitiveOps < 3)
      && { detail: `Programmatic key issued for ${e.resource}. Keys survive password resets and session revocation.` },
  },
  {
    id: 'login_profile_other',
    name: 'Console password set for another identity',
    tactic: 'Persistence',
    technique: 'T1098.001',
    severity: 4,
    risk: 82,
    // Identity administrators do this during onboarding all week. It is only
    // interesting from someone who does not routinely administer identities,
    // or under circumstances that are themselves odd.
    when: (e, ctx) => /^iam:(Create|Update)LoginProfile/i.test(e.action)
      && e.resource !== e.actor
      && (!isRoutineFor(ctx, 3) || contextStrength(e, ctx) >= 2)
      && {
        detail: `${e.actor} set a console password on ${e.resource}${!isRoutineFor(ctx, 3) ? ', and does not routinely administer identities' : `, from ${contextReason(e, ctx)}`}.`,
        risk: gradedRisk(82, contextStrength(e, ctx) + (isRoutineFor(ctx, 3) ? 0 : 2)),
      },
  },
  {
    id: 'brute_force_success',
    name: 'Successful sign-in after repeated failures',
    tactic: 'Credential Access',
    technique: 'T1110',
    severity: 5,
    risk: 88,
    when: (e, ctx) => e.outcome === 'success'
      && /(signin|SignIn|Logon|ConsoleLogin)/i.test(e.action)
      && ctx.failStreak >= 5
      && { detail: `Authentication succeeded after ${ctx.failStreak} consecutive failures — the guessing worked.` },
  },
  {
    id: 'password_spray',
    name: 'Sustained authentication failures',
    tactic: 'Credential Access',
    technique: 'T1110.003',
    severity: 3,
    risk: 62,
    when: (e, ctx) => e.outcome === 'failure'
      && /signin|SignIn|Logon/i.test(e.action)
      && ctx.failStreak >= 7
      && { detail: `${ctx.failStreak} consecutive failed authentications from ${e.ip}.` },
  },
  {
    id: 'impossible_travel',
    name: 'Impossible travel between sessions',
    tactic: 'Initial Access',
    technique: 'T1078',
    severity: 4,
    risk: 84,
    when: (e, ctx) => ctx.travelKmh > 900
      && { detail: `Implied travel of ${Math.round(ctx.travelKmh).toLocaleString()} km/h from the previous location (${e.city || e.country}). Two people, one identity.` },
  },
  {
    id: 'new_country_privileged',
    name: 'Sensitive action from a new country',
    tactic: 'Initial Access',
    technique: 'T1078.004',
    severity: 3,
    risk: 66,
    when: (e, ctx) => ctx.newCountry && ctx.sensitivity >= 0.55
      && { detail: `First activity from ${e.country || 'an unseen location'}, carrying out ${e.action}.` },
  },
  {
    id: 'enumeration_burst',
    name: 'Estate enumeration',
    tactic: 'Discovery',
    technique: 'T1580',
    severity: 3,
    risk: 64,
    when: (e, ctx) => ctx.isDiscovery
      && ctx.distinctActions1h >= 9
      && ctx.discoveryRatio >= 0.75
      && ctx.burst1h >= 14
      && { detail: `${ctx.distinctActions1h} distinct read operations in the past hour (${Math.round(ctx.discoveryRatio * 100)}% enumeration). Mapping before moving.` },
  },
  {
    id: 'secret_sweep',
    name: 'Bulk secret retrieval',
    tactic: 'Credential Access',
    technique: 'T1552.005',
    severity: 4,
    risk: 84,
    when: (e, ctx) => /^(secretsmanager:GetSecretValue|ssm:GetParameters?|kms:Decrypt)/i.test(e.action)
      && windowCount(ctx, /^(secretsmanager:GetSecretValue|ssm:GetParameters?)/i) >= 4
      && { detail: `${windowCount(ctx, /^(secretsmanager:GetSecretValue|ssm:GetParameters?)/i) + 1} secrets pulled in one hour.` },
  },
  {
    id: 'pam_bulk_retrieval',
    name: 'Bulk privileged credential checkout',
    tactic: 'Credential Access',
    technique: 'T1555',
    severity: 4,
    risk: 86,
    when: (e, ctx) => /^cyberark:RetrievePassword/i.test(e.action)
      && windowCount(ctx, /^cyberark:RetrievePassword/i) >= 3
      && { detail: `${windowCount(ctx, /^cyberark:RetrievePassword/i) + 1} vault credentials checked out within the hour by ${e.actor}.` },
  },
  {
    id: 'pam_safe_membership',
    name: 'Vault safe membership changed',
    tactic: 'Persistence',
    technique: 'T1098',
    severity: 4,
    risk: 80,
    when: (e) => /^cyberark:(Add|Update|Delete)SafeMember/i.test(e.action)
      && { detail: `Safe membership modified: ${e.resource}. Grants durable access to privileged credentials.` },
  },
  {
    id: 'pam_off_hours',
    name: 'Privileged checkout outside business hours',
    tactic: 'Credential Access',
    technique: 'T1555',
    severity: 3,
    risk: 58,
    when: (e, ctx) => /^cyberark:(RetrievePassword|Connect)/i.test(e.action)
      && (ctx.hour < 7 || ctx.hour >= 19)
      && { detail: `Vault access at ${String(ctx.hour).padStart(2, '0')}:00 UTC with no change window in progress.` },
  },
  {
    id: 'mass_object_read',
    name: 'Bulk object retrieval',
    tactic: 'Collection',
    technique: 'T1530',
    severity: 4,
    risk: 76,
    when: (e, ctx) => /^s3:GetObject/i.test(e.action)
      && ctx.burst1h >= 40
      && windowCount(ctx, /^s3:GetObject/i) >= 25
      && { detail: `${ctx.burst1h} object reads in the past hour against ${e.resource.split('/')[0]}.` },
  },
  {
    id: 'snapshot_shared',
    name: 'Disk image or snapshot shared',
    tactic: 'Exfiltration',
    technique: 'T1537',
    severity: 5,
    risk: 92,
    when: (e) => /^(ec2:(ModifySnapshotAttribute|ModifyImageAttribute)|rds:ModifyDBSnapshotAttribute)/i.test(e.action)
      && { detail: `${e.action} on ${e.resource} — data leaves the account without a single byte crossing the network boundary you monitor.` },
  },
  {
    id: 'bucket_exposed',
    name: 'Bucket policy or ACL loosened',
    tactic: 'Exfiltration',
    technique: 'T1567',
    severity: 4,
    risk: 82,
    when: (e) => /^s3:(PutBucketPolicy|PutBucketAcl|DeleteBucketPolicy|PutObjectAcl)/i.test(e.action)
      && { detail: `Access policy changed on ${e.resource}.` },
  },
  {
    id: 'sg_world_open',
    name: 'Security group opened to the internet',
    tactic: 'Persistence',
    technique: 'T1562.007',
    severity: 4,
    risk: 78,
    when: (e) => /^ec2:AuthorizeSecurityGroupIngress/i.test(e.action)
      && /0\.0\.0\.0\/0|::\/0/.test(`${e.resource} ${rawText(e)}`)
      && { detail: `Ingress opened to 0.0.0.0/0 on ${e.resource}.` },
  },
  {
    id: 'dormant_revival',
    name: 'Dormant identity reactivated',
    tactic: 'Initial Access',
    technique: 'T1078.004',
    severity: 3,
    risk: 60,
    when: (e, ctx) => ctx.dormantRevival && ctx.actorEventsSoFar >= 3
      && { detail: `${e.actor} was silent for ${Math.round(ctx.gapSec / 86400)} days before this event.` },
  },
  {
    id: 'root_usage',
    name: 'Root account activity',
    tactic: 'Privilege Escalation',
    technique: 'T1078.004',
    severity: 4,
    risk: 78,
    when: (e) => e.actorType === 'Root'
      && { detail: 'Root credentials were used. Root should be sealed after account setup.' },
  },
  {
    id: 'no_mfa_privileged',
    name: 'Privileged action after MFA drop-off',
    tactic: 'Defense Evasion',
    technique: 'T1078',
    severity: 3,
    risk: 62,
    // An identity that *never* uses MFA is a standing posture problem, surfaced
    // on its profile rather than re-alerted on every event it produces. The
    // detection here is the drop-off: usually protected, suddenly not.
    when: (e, ctx) => ctx.mfaDropOff
      && ctx.sensitivity >= 0.7
      && { detail: `${e.action} on a session with no second factor, though ${e.actor} normally presents one (${Math.round((ctx.mfaCoverage ?? 0) * 100)}% of prior sessions).` },
  },
  {
    id: 'denied_privileged',
    name: 'Denied attempt at a sensitive operation',
    tactic: 'Discovery',
    technique: 'T1078',
    severity: 3,
    risk: 58,
    when: (e, ctx) => e.outcome === 'failure' && ctx.sensitivity >= 0.8
      && { detail: `${e.action} denied (${e.errorCode || 'AccessDenied'}) — permission boundaries being probed.` },
  },
  {
    id: 'off_hours_privileged',
    name: 'Privileged change outside business hours',
    tactic: 'Defense Evasion',
    technique: 'T1078',
    severity: 2,
    risk: 52,
    when: (e, ctx) => (ctx.hour < 6 || ctx.hour >= 21) && ctx.sensitivity >= 0.75 && ctx.isWrite
      && { detail: `State-changing privileged call at ${String(ctx.hour).padStart(2, '0')}:00 UTC.` },
  },
  {
    id: 'client_shift',
    name: 'Automation on an interactive identity',
    tactic: 'Execution',
    technique: 'T1059',
    severity: 2,
    risk: 50,
    when: (e, ctx) => e.actorType === 'Human'
      && ['script', 'sdk'].includes(ctx.clientClass)
      && ctx.newUa
      && ctx.sensitivity >= 0.5
      && { detail: `Identity that normally uses a console appeared via ${ctx.clientClass} client "${String(e.userAgent).slice(0, 48)}".` },
  },
  {
    id: 'service_interactive',
    name: 'Service account used interactively',
    tactic: 'Lateral Movement',
    technique: 'T1078.004',
    severity: 4,
    risk: 76,
    when: (e, ctx) => e.actorType === 'Service'
      && (['console', 'browser'].includes(ctx.clientClass) || /signin|ConsoleLogin/i.test(e.action))
      && { detail: `${e.actor} is a non-human identity but signed in interactively.` },
  },
  {
    id: 'human_automation_burst',
    name: 'Machine-speed activity from a human identity',
    tactic: 'Execution',
    technique: 'T1059',
    severity: 3,
    risk: 62,
    when: (e, ctx) => e.actorType === 'Human' && ctx.burst1m >= 25
      && { detail: `${ctx.burst1m} calls in sixty seconds — a person is not typing this.` },
  },
  {
    id: 'destructive_burst',
    name: 'Rapid destructive operations',
    tactic: 'Impact',
    technique: 'T1485',
    severity: 4,
    risk: 80,
    when: (e, ctx) => /(Terminate|Delete|Destroy|Remove)/i.test(e.action)
      && windowCount(ctx, /(Terminate|Delete|Destroy|Remove)/i) >= 4
      && ctx.burst5m >= 5
      && { detail: `${windowCount(ctx, /(Terminate|Delete|Destroy|Remove)/i) + 1} destructive calls in quick succession.` },
  },
  {
    id: 'key_pair_created',
    name: 'Compute key pair created',
    tactic: 'Persistence',
    technique: 'T1098.004',
    severity: 3,
    risk: 60,
    when: (e, ctx) => /^ec2:CreateKeyPair/i.test(e.action) && (ctx.newAction || ctx.newIp)
      && { detail: `SSH key pair "${e.resource}" created from ${ctx.newIp ? 'an unfamiliar address' : 'this identity for the first time'}.` },
  },
];

export const RULE_INDEX = Object.fromEntries(RULES.map((r) => [r.id, r]));

/**
 * Evaluate every rule against one event.
 * @returns {Array<{id,name,tactic,technique,severity,risk,detail}>}
 */
export function evaluateRules(event, ctx, enabled = null) {
  const hits = [];
  for (const rule of RULES) {
    if (enabled && !enabled.has(rule.id)) continue;
    let result;
    try {
      result = rule.when(event, ctx);
    } catch {
      continue;
    }
    if (result) {
      const graded = typeof result === 'object' && typeof result.risk === 'number' ? result.risk : rule.risk;
      hits.push({
        id: rule.id,
        name: rule.name,
        tactic: rule.tactic,
        technique: rule.technique,
        // A rule whose risk was graded down by weak context should not keep an
        // escalating severity, or it would bypass the threshold anyway.
        severity: graded < rule.risk * 0.85 ? Math.max(2, rule.severity - 1) : rule.severity,
        risk: graded,
        detail: typeof result === 'object' && result.detail ? result.detail : `${rule.name} matched.`,
      });
    }
  }
  return hits;
}
