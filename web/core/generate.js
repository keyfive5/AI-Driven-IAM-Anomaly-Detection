/**
 * Synthetic IAM corpus generator with ground truth.
 *
 * The original project evaluated on a generator that emitted uniformly random
 * events, which makes "anomaly" trivially separable and the resulting metrics
 * meaningless. This generator instead models the two things that make identity
 * telemetry hard:
 *
 *   1. Normal is structured, not uniform. Every principal has a role, a home
 *      network, working hours, a repertoire of API calls it actually uses, and
 *      a weekly rhythm. Service accounts behave nothing like humans.
 *
 *   2. Some benign behaviour looks weird. Travel, on-call pages at 03:00,
 *      onboarding bursts, and a quarterly access review all produce events that
 *      a naive detector flags. They are emitted with label 0 on purpose, so
 *      precision has to be earned rather than assumed.
 *
 * Attacks are injected as multi-stage campaigns with per-event ground truth,
 * which is what lets the Evaluate view report real precision/recall and
 * time-to-detect instead of quoted numbers.
 */

import { makeEvent, GEO } from './parse.js';
import { makeRng } from './rng.js';

const MIN = 60_000;
const HOUR = 3_600_000;
const DAY = 86_400_000;

const FIRST = ['james', 'maria', 'wei', 'aisha', 'daniel', 'priya', 'omar', 'sofia', 'liam', 'nina',
  'raj', 'elena', 'tom', 'grace', 'hassan', 'yuki', 'pablo', 'ivy', 'noah', 'zara',
  'kwame', 'lena', 'victor', 'mei', 'jonas', 'farah', 'ravi', 'clara', 'diego', 'anna'];
const LAST = ['smith', 'chen', 'rivera', 'okafor', 'novak', 'patel', 'haddad', 'martins', 'oconnor', 'kim',
  'singh', 'petrova', 'brooks', 'adeyemi', 'zafar', 'tanaka', 'gomez', 'nguyen', 'walsh', 'ahmadi',
  'mensah', 'kowalski', 'silva', 'zhang', 'berg', 'khalil', 'iyer', 'dubois', 'ortiz', 'lindqvist'];

/**
 * Job families: what a principal legitimately does all day.
 *
 * `flows` matter as much as `actions`. Real operators work in short task
 * sequences — assume a role, look at instances, tail the logs — and it is that
 * sequential regularity a Markov baseline learns. A generator that emits
 * independent draws from a weighted bag produces traffic with no sequence
 * structure at all, which makes any temporal model look worthless for reasons
 * that have nothing to do with the model.
 */
const TEAMS = {
  platform: {
    weight: 0.22,
    actions: [
      ['ec2:DescribeInstances', 10], ['ec2:RunInstances', 2], ['ec2:TerminateInstances', 1.2],
      ['ec2:DescribeSecurityGroups', 4], ['ec2:AuthorizeSecurityGroupIngress', 1],
      ['s3:GetObject', 6], ['s3:PutObject', 4], ['s3:ListBucket', 5],
      ['cloudformation:DescribeStacks', 4], ['cloudformation:UpdateStack', 1.5],
      ['logs:FilterLogEvents', 5], ['sts:AssumeRole', 6], ['signin:ConsoleLogin', 1.5],
      ['lambda:UpdateFunctionCode', 2], ['lambda:Invoke', 3],
    ],
    flows: [
      ['sts:AssumeRole', 'ec2:DescribeInstances', 'logs:FilterLogEvents'],
      ['signin:ConsoleLogin', 'ec2:DescribeInstances', 'ec2:DescribeSecurityGroups'],
      ['cloudformation:DescribeStacks', 'cloudformation:UpdateStack', 'ec2:DescribeInstances'],
      ['s3:ListBucket', 's3:GetObject', 's3:PutObject'],
      ['lambda:UpdateFunctionCode', 'lambda:Invoke', 'logs:FilterLogEvents'],
      ['sts:AssumeRole', 'ec2:RunInstances', 'ec2:DescribeInstances'],
    ],
    resources: ['prod-api', 'prod-worker', 'staging-api', 'edge-cache', 'build-runner', 'artifacts-bucket'],
    clients: ['aws-cli/2.15.2 Python/3.11.6 Linux/5.15', 'Terraform/1.7.2 (+https://www.terraform.io) aws-sdk-go/1.49',
      'Boto3/1.34.20 Python/3.11.6', 'console.amazonaws.com'],
  },
  data: {
    weight: 0.18,
    actions: [
      ['s3:GetObject', 14], ['s3:ListBucket', 8], ['s3:PutObject', 5],
      ['glue:StartJobRun', 3], ['athena:StartQueryExecution', 6],
      ['rds:DescribeDBInstances', 3], ['kms:Decrypt', 4], ['sts:AssumeRole', 4],
      ['signin:ConsoleLogin', 1.2], ['secretsmanager:GetSecretValue', 1.5],
    ],
    flows: [
      ['sts:AssumeRole', 's3:ListBucket', 's3:GetObject', 's3:GetObject', 'athena:StartQueryExecution'],
      ['athena:StartQueryExecution', 's3:GetObject', 's3:PutObject'],
      ['kms:Decrypt', 's3:GetObject', 's3:PutObject'],
      ['glue:StartJobRun', 'rds:DescribeDBInstances', 's3:PutObject'],
      ['signin:ConsoleLogin', 'athena:StartQueryExecution', 's3:GetObject'],
    ],
    resources: ['analytics-lake', 'events-raw', 'reporting-db', 'ml-features', 'finance-extract'],
    clients: ['Boto3/1.34.20 Python/3.11.6', 'aws-cli/2.15.2 Python/3.11.6 Linux/5.15', 'console.amazonaws.com'],
  },
  security: {
    weight: 0.1,
    actions: [
      ['iam:ListUsers', 6], ['iam:ListRoles', 5], ['iam:GetAccountAuthorizationDetails', 2],
      ['guardduty:ListFindings', 5], ['cloudtrail:LookupEvents', 6], ['config:DescribeRules', 3],
      ['iam:ListAccessKeys', 3], ['signin:ConsoleLogin', 2], ['sts:AssumeRole', 3],
      ['iam:GenerateCredentialReport', 1],
    ],
    flows: [
      ['signin:ConsoleLogin', 'guardduty:ListFindings', 'cloudtrail:LookupEvents'],
      ['iam:ListUsers', 'iam:ListAccessKeys', 'iam:GetAccountAuthorizationDetails'],
      ['cloudtrail:LookupEvents', 'ec2:DescribeInstances', 'guardduty:ListFindings'],
      ['sts:AssumeRole', 'config:DescribeRules', 'cloudtrail:LookupEvents'],
    ],
    resources: ['audit-account', 'org-root', 'security-hub', 'flow-logs'],
    clients: ['console.amazonaws.com', 'aws-cli/2.15.2 Python/3.11.6 Darwin/23.2'],
  },
  finance: {
    weight: 0.14,
    actions: [
      ['signin:ConsoleLogin', 3], ['ce:GetCostAndUsage', 8], ['s3:GetObject', 4],
      ['budgets:DescribeBudgets', 3], ['organizations:ListAccounts', 2],
    ],
    flows: [
      ['signin:ConsoleLogin', 'ce:GetCostAndUsage', 'budgets:DescribeBudgets'],
      ['ce:GetCostAndUsage', 's3:GetObject'],
      ['signin:ConsoleLogin', 'organizations:ListAccounts', 'ce:GetCostAndUsage'],
    ],
    resources: ['billing-reports', 'cost-explorer', 'invoices-2026'],
    clients: ['console.amazonaws.com', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0'],
  },
  support: {
    weight: 0.16,
    actions: [
      ['signin:ConsoleLogin', 3], ['logs:FilterLogEvents', 8], ['s3:GetObject', 4],
      ['ec2:DescribeInstances', 4], ['rds:DescribeDBInstances', 2], ['sts:AssumeRole', 3],
      ['ssm:StartSession', 2],
    ],
    flows: [
      ['signin:ConsoleLogin', 'logs:FilterLogEvents', 'ec2:DescribeInstances'],
      ['ssm:StartSession', 'logs:FilterLogEvents'],
      ['sts:AssumeRole', 'rds:DescribeDBInstances', 'logs:FilterLogEvents'],
      ['s3:GetObject', 'logs:FilterLogEvents'],
    ],
    resources: ['support-tools', 'prod-api', 'customer-exports'],
    clients: ['console.amazonaws.com', 'aws-cli/2.15.2 Python/3.11.6 Windows/10'],
  },
  admin: {
    weight: 0.06,
    actions: [
      ['iam:ListUsers', 5], ['iam:CreateUser', 1], ['iam:AddUserToGroup', 1],
      ['iam:AttachUserPolicy', 0.8], ['iam:ListRoles', 3], ['iam:CreateRole', 0.7],
      ['signin:ConsoleLogin', 3], ['organizations:ListAccounts', 2], ['sts:AssumeRole', 3],
      ['iam:DeleteUser', 0.4], ['iam:ListAccessKeys', 2],
    ],
    // Identity administration is this team's day job — onboarding runs as a
    // routine sequence, which is precisely why "someone called CreateUser"
    // cannot be an alert on its own.
    flows: [
      ['signin:ConsoleLogin', 'iam:ListUsers', 'iam:ListAccessKeys'],
      ['iam:CreateUser', 'iam:AddUserToGroup', 'iam:AttachUserPolicy'],
      ['sts:AssumeRole', 'organizations:ListAccounts', 'iam:ListRoles'],
      ['iam:ListUsers', 'iam:DeleteUser'],
      ['iam:CreateRole', 'iam:ListRoles'],
    ],
    resources: ['org-root', 'identity-center', 'break-glass'],
    clients: ['console.amazonaws.com', 'aws-cli/2.15.2 Python/3.11.6 Darwin/23.2'],
  },
  contractor: {
    weight: 0.14,
    actions: [
      ['signin:ConsoleLogin', 2], ['s3:GetObject', 6], ['s3:ListBucket', 4],
      ['ec2:DescribeInstances', 3], ['logs:FilterLogEvents', 3],
    ],
    flows: [
      ['signin:ConsoleLogin', 's3:ListBucket', 's3:GetObject'],
      ['ec2:DescribeInstances', 'logs:FilterLogEvents'],
      ['s3:ListBucket', 's3:GetObject', 's3:GetObject'],
    ],
    resources: ['vendor-share', 'staging-api', 'docs-bucket'],
    clients: ['console.amazonaws.com', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/121.0'],
  },
};

/** Non-human identities run a fixed cycle — the most predictable traffic there is. */
const SERVICE_ACCOUNTS = [
  { name: 'svc-billing-etl', cycle: ['s3:ListBucket', 's3:GetObject', 'kms:Decrypt', 's3:PutObject', 'glue:StartJobRun'], every: 15 * MIN },
  { name: 'svc-backup-runner', cycle: ['ec2:DescribeVolumes', 'ec2:CreateSnapshot', 'rds:CreateDBSnapshot'], every: 30 * MIN },
  { name: 'svc-ci-deployer', cycle: ['sts:AssumeRole', 's3:PutObject', 'lambda:UpdateFunctionCode', 'cloudformation:UpdateStack'], every: 12 * MIN },
  { name: 'svc-log-shipper', cycle: ['logs:PutLogEvents', 'logs:PutLogEvents', 's3:PutObject'], every: 8 * MIN },
  { name: 'svc-config-scanner', cycle: ['config:DescribeRules', 'ec2:DescribeInstances', 'iam:ListUsers'], every: 60 * MIN },
];

const OFFICE_SITES = ['CA-ON', 'US-VA', 'US-CA', 'IE-D', 'GB-LND'];
const HOSTILE_SITES = ['RU-MOW', 'NG-LA', 'CN-BJ', 'BR-SP'];

const REGION_BY_SITE = {
  'CA-ON': 'ca-central-1', 'US-VA': 'us-east-1', 'US-CA': 'us-west-2', 'IE-D': 'eu-west-1',
  'GB-LND': 'eu-west-2', 'DE-HE': 'eu-central-1', 'SG': 'ap-southeast-1', 'IN-MH': 'ap-south-1',
  'BR-SP': 'sa-east-1', 'RU-MOW': 'eu-central-1', 'NG-LA': 'eu-west-1', 'CN-BJ': 'ap-southeast-1',
  'NL-NH': 'eu-west-1',
};

function siteEvent(site, extra = {}) {
  const g = GEO[site];
  return {
    region: REGION_BY_SITE[site] || 'us-east-1',
    country: g?.country ?? null,
    city: g?.city ?? null,
    lat: g?.lat ?? null,
    lon: g?.lon ?? null,
    ...extra,
  };
}

/** Build the population of principals. */
function buildPersonas(rng, count) {
  const teamNames = Object.keys(TEAMS);
  const weights = teamNames.map((t) => TEAMS[t].weight);
  const used = new Set();
  const personas = [];

  for (let i = 0; i < count; i++) {
    const team = rng.weighted(teamNames, weights);
    let handle;
    do {
      handle = `${rng.pick(FIRST)[0]}${rng.pick(LAST)}`;
    } while (used.has(handle));
    used.add(handle);

    const site = team === 'contractor' ? rng.pick(['IN-MH', 'BR-SP', 'CA-ON', 'IE-D']) : rng.pick(OFFICE_SITES);
    const startHour = rng.intRange(7, 10);
    personas.push({
      id: handle,
      team,
      site,
      actorType: 'Human',
      startHour,
      endHour: startHour + rng.intRange(8, 10),
      ip: `${rng.intRange(11, 198)}.${rng.intRange(0, 255)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`,
      altIp: `10.${rng.intRange(0, 40)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`,
      client: rng.pick(TEAMS[team].clients),
      volume: Math.max(4, Math.round(rng.normal(team === 'admin' ? 26 : 34, 12))),
      mfa: rng.bool(0.85),
      dormant: rng.bool(0.06),
    });
  }

  // Every organisation has someone who administers identities. Drawing the team
  // purely at random occasionally produces an estate with no administrator at
  // all, which quietly removes a whole class of behaviour from the corpus.
  const admins = personas.filter((p) => p.team === 'admin');
  for (let i = admins.length; i < 2 && i < personas.length; i++) {
    const victim = personas.find((p) => p.team !== 'admin');
    if (!victim) break;
    victim.team = 'admin';
    victim.client = rng.pick(TEAMS.admin.clients);
  }

  for (const svc of SERVICE_ACCOUNTS) {
    personas.push({
      id: svc.name,
      team: 'service',
      site: 'US-VA',
      actorType: 'Service',
      startHour: 0,
      endHour: 24,
      ip: `10.20.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`,
      altIp: null,
      client: 'Boto3/1.34.20 Python/3.11.6 exec-env/AWS_Lambda',
      volume: 0,
      cadence: svc.every,
      cycle: svc.cycle,
      actions: svc.cycle.map((a) => [a, 1]),
      mfa: null,
      dormant: false,
    });
  }
  return personas;
}

function actionPool(persona) {
  const spec = persona.actions || TEAMS[persona.team].actions;
  return { items: spec.map((a) => a[0]), weights: spec.map((a) => a[1]) };
}

function resourceFor(rng, persona) {
  const list = TEAMS[persona.team]?.resources || ['pipeline', 'bucket', 'queue'];
  return rng.pick(list);
}

/** Draw an hour-of-day for a human, mostly inside their working window. */
function drawHour(rng, persona) {
  if (rng.bool(0.06)) return rng.intRange(0, 23); // evening catch-up, early start
  const mid = (persona.startHour + persona.endHour) / 2;
  const h = Math.round(rng.normal(mid, (persona.endHour - persona.startHour) / 4.5));
  return Math.min(23, Math.max(0, h));
}

function baseEvent(rng, persona, ts, action, opts = {}) {
  const failed = opts.outcome === 'failure' || (opts.outcome === undefined && rng.bool(0.035));
  return makeEvent({
    ts,
    actor: persona.id,
    actorType: persona.actorType,
    action,
    resource: opts.resource || resourceFor(rng, persona),
    ip: opts.ip || persona.ip,
    userAgent: opts.userAgent || persona.client,
    outcome: failed ? 'failure' : 'success',
    errorCode: failed ? (opts.errorCode || 'AccessDenied') : null,
    mfa: opts.mfa !== undefined ? opts.mfa : persona.mfa,
    sessionId: opts.sessionId || null,
    source: 'synthetic',
    label: opts.label ? 1 : 0,
    campaign: opts.campaign || null,
    stage: opts.stage || null,
    ...siteEvent(opts.site || persona.site),
  });
}

/* ------------------------------------------------------------ campaigns --- */

/**
 * Each campaign returns { name, tactic, technique, events } where every event
 * carries label 1 plus the kill-chain stage that produced it. Campaigns are
 * written to be individually plausible: the noisy ones are easy, the quiet ones
 * are the reason ensemble scoring exists.
 */
const CAMPAIGNS = {
  credential_stuffing: {
    name: 'Password spray → account takeover',
    technique: 'T1110.003 Password Spraying',
    difficulty: 'loud',
    build(rng, ctx) {
      const victim = ctx.pickHuman((p) => p.team !== 'service');
      const site = rng.pick(HOSTILE_SITES);
      const attackerIp = `${rng.intRange(41, 203)}.${rng.intRange(0, 255)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`;
      const ua = 'python-requests/2.31.0';
      const start = ctx.timeAt(rng.intRange(1, ctx.days - 1), rng.intRange(1, 4));
      const events = [];

      for (let i = 0; i < rng.intRange(14, 26); i++) {
        events.push(baseEvent(rng, victim, start + i * rng.intRange(4, 20) * 1000, 'signin:ConsoleLogin', {
          ip: attackerIp, userAgent: ua, site, outcome: 'failure', errorCode: 'Failed authentication',
          mfa: false, resource: 'aws-console', stage: 'Credential Access', label: 1,
        }));
      }
      const successTs = events[events.length - 1].ts + 30_000;
      events.push(baseEvent(rng, victim, successTs, 'signin:ConsoleLogin', {
        ip: attackerIp, userAgent: ua, site, outcome: 'success', mfa: false,
        resource: 'aws-console', stage: 'Initial Access', label: 1,
      }));
      events.push(baseEvent(rng, victim, successTs + 3 * MIN, 'iam:DeactivateMFADevice', {
        ip: attackerIp, userAgent: ua, site, resource: victim.id, stage: 'Defense Evasion', label: 1,
      }));
      events.push(baseEvent(rng, victim, successTs + 6 * MIN, 'iam:CreateAccessKey', {
        ip: attackerIp, userAgent: ua, site, resource: victim.id, stage: 'Persistence', label: 1,
      }));
      return { actor: victim.id, events };
    },
  },

  privilege_escalation: {
    name: 'Self-service privilege escalation',
    technique: 'T1098 Account Manipulation',
    difficulty: 'medium',
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.team === 'platform' || p.team === 'data');
      const start = ctx.timeAt(rng.intRange(2, ctx.days - 1), rng.intRange(19, 22));
      const events = [];
      const ip = actor.ip;
      const seqActions = [
        ['iam:GetAccountAuthorizationDetails', 'Discovery'],
        ['iam:ListRoles', 'Discovery'],
        ['iam:ListAttachedUserPolicies', 'Discovery'],
        ['iam:ListPolicies', 'Discovery'],
      ];
      let t = start;
      for (const [action, stage] of seqActions) {
        events.push(baseEvent(rng, actor, t, action, { ip, resource: 'org-root', stage, label: 1, outcome: 'success' }));
        t += rng.intRange(20, 90) * 1000;
      }
      events.push(baseEvent(rng, actor, t, 'iam:AttachUserPolicy', {
        ip, resource: 'arn:aws:iam::aws:policy/AdministratorAccess', stage: 'Privilege Escalation',
        label: 1, outcome: 'failure', errorCode: 'AccessDenied',
      }));
      t += 4 * MIN;
      events.push(baseEvent(rng, actor, t, 'sts:AssumeRole', {
        ip, resource: 'DeploymentAdminRole', stage: 'Privilege Escalation', label: 1,
      }));
      t += 90 * 1000;
      events.push(baseEvent(rng, actor, t, 'iam:AttachUserPolicy', {
        ip, resource: 'arn:aws:iam::aws:policy/AdministratorAccess', stage: 'Privilege Escalation', label: 1,
      }));
      t += 2 * MIN;
      events.push(baseEvent(rng, actor, t, 'iam:CreateLoginProfile', {
        ip, resource: 'break-glass-2', stage: 'Persistence', label: 1,
      }));
      return { actor: actor.id, events };
    },
  },

  persistence_evasion: {
    name: 'Backdoor identity + audit tamper',
    technique: 'T1562.008 Disable Cloud Logs',
    difficulty: 'medium',
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.team === 'admin' || p.team === 'platform');
      const start = ctx.timeAt(rng.intRange(3, ctx.days - 1), rng.intRange(2, 5));
      const ip = `${rng.intRange(41, 203)}.${rng.intRange(0, 255)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`;
      const site = rng.pick(HOSTILE_SITES);
      const ua = 'aws-cli/2.15.2 Python/3.11.6 Linux/5.15';
      const events = [];
      let t = start;
      const steps = [
        ['iam:CreateUser', 'Persistence', 'svc-monitoring-agent'],
        ['iam:CreateAccessKey', 'Persistence', 'svc-monitoring-agent'],
        ['iam:AttachUserPolicy', 'Privilege Escalation', 'arn:aws:iam::aws:policy/AdministratorAccess'],
        ['cloudtrail:StopLogging', 'Defense Evasion', 'org-trail'],
        ['guardduty:DeleteDetector', 'Defense Evasion', 'gd-detector-1'],
        ['cloudtrail:DeleteTrail', 'Defense Evasion', 'org-trail'],
      ];
      for (const [action, stage, resource] of steps) {
        events.push(baseEvent(rng, actor, t, action, { ip, userAgent: ua, site, resource, stage, label: 1, mfa: false }));
        t += rng.intRange(40, 200) * 1000;
      }
      return { actor: actor.id, events };
    },
  },

  data_exfiltration: {
    name: 'Bulk data staging and snapshot sharing',
    technique: 'T1537 Transfer Data to Cloud Account',
    difficulty: 'medium',
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.team === 'data' || p.team === 'support');
      const start = ctx.timeAt(rng.intRange(2, ctx.days - 1), 3);
      const ip = actor.altIp || actor.ip;
      const events = [];
      let t = start;
      events.push(baseEvent(rng, actor, t, 's3:ListBucket', { ip, resource: 'customer-pii', stage: 'Collection', label: 1 }));
      t += 30_000;
      for (let i = 0; i < rng.intRange(45, 80); i++) {
        events.push(baseEvent(rng, actor, t, 's3:GetObject', {
          ip, resource: `customer-pii/export-${i}.parquet`, stage: 'Collection', label: 1, outcome: 'success',
        }));
        t += rng.intRange(2, 9) * 1000;
      }
      events.push(baseEvent(rng, actor, t, 'ec2:CreateSnapshot', { ip, resource: 'vol-prod-db', stage: 'Collection', label: 1 }));
      t += 3 * MIN;
      events.push(baseEvent(rng, actor, t, 'ec2:ModifySnapshotAttribute', {
        ip, resource: 'snap-prod-db (shared: 999988887777)', stage: 'Exfiltration', label: 1,
      }));
      return { actor: actor.id, events };
    },
  },

  insider_recon: {
    name: 'Dormant contractor account revival',
    technique: 'T1078.004 Valid Cloud Accounts',
    difficulty: 'quiet',
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.team === 'contractor') || ctx.pickHuman();
      const start = ctx.timeAt(ctx.days - 2, rng.intRange(21, 23));
      const ip = `${rng.intRange(41, 203)}.${rng.intRange(0, 255)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`;
      const site = rng.pick(['NL-NH', 'SG', 'RU-MOW']);
      const events = [];
      let t = start;
      events.push(baseEvent(rng, actor, t, 'signin:ConsoleLogin', {
        ip, site, resource: 'aws-console', mfa: false, stage: 'Initial Access', label: 1,
      }));
      t += 2 * MIN;
      const recon = ['iam:ListUsers', 'iam:ListRoles', 'ec2:DescribeInstances', 's3:ListAllMyBuckets',
        'rds:DescribeDBInstances', 'secretsmanager:ListSecrets', 'ssm:DescribeParameters',
        'organizations:ListAccounts', 'iam:ListAccessKeys'];
      for (const action of recon) {
        events.push(baseEvent(rng, actor, t, action, { ip, site, resource: 'org-root', stage: 'Discovery', label: 1 }));
        t += rng.intRange(15, 60) * 1000;
      }
      for (let i = 0; i < 6; i++) {
        events.push(baseEvent(rng, actor, t, 'secretsmanager:GetSecretValue', {
          ip, site, resource: `prod/db/credential-${i}`, stage: 'Credential Access', label: 1,
        }));
        t += rng.intRange(10, 40) * 1000;
      }
      return { actor: actor.id, events };
    },
  },

  impossible_travel: {
    name: 'Impossible travel session hijack',
    technique: 'T1078 Valid Accounts',
    difficulty: 'quiet',
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.site === 'CA-ON' || p.site === 'US-VA') || ctx.pickHuman();
      const start = ctx.timeAt(rng.intRange(2, ctx.days - 1), rng.intRange(10, 15));
      const events = [];
      events.push(baseEvent(rng, actor, start, 'signin:ConsoleLogin', {
        resource: 'aws-console', stage: 'Initial Access', label: 0,
      }));
      const hijackSite = 'SG';
      const ip = `${rng.intRange(101, 203)}.${rng.intRange(0, 255)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`;
      let t = start + rng.intRange(16, 28) * MIN;
      events.push(baseEvent(rng, actor, t, 'signin:ConsoleLogin', {
        ip, site: hijackSite, resource: 'aws-console', mfa: false, stage: 'Initial Access', label: 1,
      }));
      t += 4 * MIN;
      events.push(baseEvent(rng, actor, t, 'iam:CreateAccessKey', {
        ip, site: hijackSite, resource: actor.id, stage: 'Persistence', label: 1,
      }));
      t += 5 * MIN;
      events.push(baseEvent(rng, actor, t, 'ec2:RunInstances', {
        ip, site: hijackSite, resource: 'c5.24xlarge x8', stage: 'Impact', label: 1,
      }));
      return { actor: actor.id, events };
    },
  },

  insider_admin: {
    name: 'Trusted insider backdoor',
    technique: 'T1098.003 Additional Cloud Roles',
    difficulty: 'stealth',
    /**
     * The case that exists to defeat behavioural detection, and the reason a
     * rule pack earns its place in the ensemble.
     *
     * An identity administrator creates a user, adds it to a group and attaches
     * a policy — their exact onboarding sequence, at 14:00, from their usual
     * office address, on an MFA session, using an operation they perform every
     * week. Nothing about the *behaviour* is unusual, so every unsupervised
     * detector correctly finds it unremarkable. What is wrong is the object:
     * the policy is AdministratorAccess and the "new hire" is a service-shaped
     * name nobody hired. Only a rule that knows what a full-control grant means
     * catches this.
     */
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.team === 'admin');
      if (!actor) return null;
      const start = ctx.timeAt(rng.intRange(6, ctx.days - 2), 14);
      const backdoor = 'svc-metrics-collector';
      const events = [];
      let t = start;
      events.push(baseEvent(rng, actor, t, 'iam:CreateUser', {
        resource: backdoor, stage: 'Persistence', label: 1, mfa: true,
      }));
      t += rng.intRange(40, 120) * 1000;
      events.push(baseEvent(rng, actor, t, 'iam:AddUserToGroup', {
        resource: `${backdoor}→developers`, stage: 'Persistence', label: 1, mfa: true,
      }));
      t += rng.intRange(40, 150) * 1000;
      events.push(baseEvent(rng, actor, t, 'iam:AttachUserPolicy', {
        resource: 'arn:aws:iam::aws:policy/AdministratorAccess',
        stage: 'Privilege Escalation', label: 1, mfa: true,
      }));
      t += rng.intRange(60, 200) * 1000;
      events.push(baseEvent(rng, actor, t, 'iam:CreateAccessKey', {
        resource: backdoor, stage: 'Persistence', label: 1, mfa: true,
      }));
      return { actor: actor.id, events };
    },
  },

  pam_abuse: {
    name: 'Privileged vault abuse (PAM)',
    technique: 'T1555 Credentials from Password Stores',
    difficulty: 'quiet',
    build(rng, ctx) {
      const actor = ctx.pickHuman((p) => p.team === 'support' || p.team === 'platform');
      const start = ctx.timeAt(rng.intRange(2, ctx.days - 1), rng.intRange(23, 23));
      const events = [];
      let t = start;
      events.push(baseEvent(rng, actor, t, 'cyberark:Logon', {
        resource: 'PVWA', stage: 'Initial Access', label: 1, mfa: false,
      }));
      t += 40_000;
      for (let i = 0; i < 7; i++) {
        events.push(baseEvent(rng, actor, t, 'cyberark:RetrievePassword', {
          resource: `Safe-Prod-DB/root-account-${i}`, stage: 'Credential Access', label: 1,
        }));
        t += rng.intRange(20, 70) * 1000;
      }
      events.push(baseEvent(rng, actor, t, 'cyberark:AddSafeMember', {
        resource: 'Safe-Prod-DB/+contractor-tmp', stage: 'Persistence', label: 1,
      }));
      t += 3 * MIN;
      events.push(baseEvent(rng, actor, t, 'cyberark:Connect', {
        resource: 'prod-db-01 (session 214 min)', stage: 'Lateral Movement', label: 1,
      }));
      return { actor: actor.id, events };
    },
  },
};

/* ------------------------------------------------------ benign confusers --- */

/** Odd-looking but legitimate activity. Ground truth 0 — these cost precision. */
function benignConfusers(rng, ctx, events) {
  // 1. On-call incident response at 03:00.
  const oncall = ctx.pickHuman((p) => p.team === 'platform' || p.team === 'support');
  if (oncall) {
    let t = ctx.timeAt(rng.intRange(1, ctx.days - 1), 3);
    for (let i = 0; i < rng.intRange(12, 22); i++) {
      events.push(baseEvent(rng, oncall, t, rng.pick(['logs:FilterLogEvents', 'ec2:DescribeInstances', 'ec2:RebootInstances', 'lambda:Invoke']), {
        resource: 'prod-api', label: 0,
      }));
      t += rng.intRange(30, 120) * 1000;
    }
  }

  // 2. Genuine business travel is handled before baseline generation (see
  //    `assignTravel`) so the traveller is in one place at a time. Emitting a
  //    parallel stream from abroad while their office traffic continues would
  //    manufacture impossible-travel hits that no real trip produces.

  // 3. Quarterly access review by security: heavy IAM enumeration, all legitimate.
  const auditor = ctx.pickHuman((p) => p.team === 'security');
  if (auditor) {
    let t = ctx.timeAt(rng.intRange(1, ctx.days - 1), 10);
    const reviewActions = ['iam:ListUsers', 'iam:ListRoles', 'iam:ListAccessKeys', 'iam:GetAccountAuthorizationDetails',
      'iam:ListAttachedUserPolicies', 'iam:GenerateCredentialReport', 'organizations:ListAccounts'];
    for (let i = 0; i < rng.intRange(40, 70); i++) {
      events.push(baseEvent(rng, auditor, t, rng.pick(reviewActions), { resource: 'org-root', label: 0 }));
      t += rng.intRange(5, 30) * 1000;
    }
  }

  // 4. Onboarding week: an admin legitimately creates several identities.
  const admin = ctx.pickHuman((p) => p.team === 'admin');
  if (admin) {
    let t = ctx.timeAt(rng.intRange(1, ctx.days - 1), 11);
    for (let i = 0; i < rng.intRange(3, 6); i++) {
      const newHire = `${rng.pick(FIRST)[0]}${rng.pick(LAST)}`;
      events.push(baseEvent(rng, admin, t, 'iam:CreateUser', { resource: newHire, label: 0 }));
      events.push(baseEvent(rng, admin, t + 40_000, 'iam:AddUserToGroup', { resource: `${newHire}→developers`, label: 0 }));
      events.push(baseEvent(rng, admin, t + 90_000, 'iam:CreateLoginProfile', { resource: newHire, label: 0 }));
      t += rng.intRange(8, 25) * MIN;
    }
  }

  // 5. A service account with a deploy storm after a release.
  const svc = ctx.personas.find((p) => p.id === 'svc-ci-deployer');
  if (svc) {
    let t = ctx.timeAt(rng.intRange(1, ctx.days - 1), 16);
    for (let i = 0; i < rng.intRange(60, 110); i++) {
      events.push(baseEvent(rng, svc, t, rng.pick(['lambda:UpdateFunctionCode', 's3:PutObject', 'cloudformation:UpdateStack']), {
        resource: 'release-2026.8', label: 0,
      }));
      t += rng.intRange(3, 20) * 1000;
    }
  }
}

/* -------------------------------------------------------------- assembly --- */

export const CAMPAIGN_IDS = Object.keys(CAMPAIGNS);

export function campaignInfo(id) {
  const c = CAMPAIGNS[id];
  return c ? { id, name: c.name, technique: c.technique, difficulty: c.difficulty } : null;
}

/**
 * Generate a labelled corpus.
 *
 * @param {object} opts
 * @param {number} opts.seed        reproducibility seed
 * @param {number} opts.days        length of the observation window
 * @param {number} opts.users       number of human principals
 * @param {string[]} opts.campaigns campaign ids to inject (default: all)
 * @param {number} opts.endTs       timestamp the window ends at (default: now)
 */
export function generateCorpus(opts = {}) {
  const {
    seed = 20260813,
    days = 14,
    users = 40,
    campaigns = CAMPAIGN_IDS,
    endTs = Date.now(),
  } = opts;

  const rng = makeRng(seed);
  const personas = buildPersonas(rng, users);
  // Snap the window to midnight UTC. Anchoring days to "now" would offset every
  // persona's working hours by however far into the day the corpus happened to
  // be generated, which silently destroys the diurnal structure that off-hours
  // detection depends on.
  const endMidnight = Math.floor(endTs / DAY) * DAY;
  const startTs = endMidnight - days * DAY;
  const dayStart = (d) => startTs + d * DAY;

  const ctx = {
    personas,
    days,
    timeAt: (day, hour) => dayStart(Math.min(days - 1, Math.max(0, day))) + hour * HOUR + Math.floor(rng() * HOUR),
    pickHuman: (pred) => {
      const pool = personas.filter((p) => p.actorType === 'Human' && !p.dormant && (!pred || pred(p)));
      return pool.length ? rng.pick(pool) : null;
    },
  };

  const events = [];

  // --- Business travel ------------------------------------------------------
  // Chosen up front so the baseline loop can relocate the traveller for those
  // days rather than running a second, overlapping stream.
  const traveller = ctx.pickHuman((p) => p.team !== 'service' && p.team !== 'admin');
  if (traveller) {
    traveller.travelFrom = rng.intRange(2, Math.max(3, days - 4));
    traveller.travelDays = rng.intRange(2, 4);
    traveller.travelSite = rng.pick(['DE-HE', 'GB-LND', 'SG']);
    traveller.travelIp = `${rng.intRange(41, 203)}.${rng.intRange(0, 255)}.${rng.intRange(0, 255)}.${rng.intRange(2, 250)}`;
  }

  // --- Baseline traffic -----------------------------------------------------
  for (const persona of personas) {
    if (persona.actorType === 'Service') {
      // Fixed cycle at a fixed cadence, with light jitter. Boring on purpose.
      let step = 0;
      for (let t = startTs; t < endMidnight; t += persona.cadence * rng.range(0.75, 1.25)) {
        const action = persona.cycle[step % persona.cycle.length];
        step++;
        events.push(baseEvent(rng, persona, Math.round(t), action, {
          resource: rng.pick(['analytics-lake', 'prod-api', 'artifacts-bucket', 'events-raw']),
          outcome: rng.bool(0.012) ? 'failure' : 'success',
        }));
      }
      continue;
    }

    const { items, weights } = actionPool(persona);
    const flows = TEAMS[persona.team].flows;
    for (let d = 0; d < days; d++) {
      const date = new Date(dayStart(d));
      const dow = date.getUTCDay();
      const weekend = dow === 0 || dow === 6;
      if (persona.dormant && d < days - 3) continue;
      let target = Math.round(persona.volume * rng.range(0.6, 1.4) * (weekend ? 0.12 : 1));
      if (persona.dormant) target = Math.round(target * 0.15);

      // Away on business: same person, same work, different continent.
      const away = persona.travelFrom !== undefined
        && d >= persona.travelFrom
        && d < persona.travelFrom + persona.travelDays;
      const site = away ? persona.travelSite : persona.site;
      const homeIp = away ? persona.travelIp : persona.ip;

      let emitted = 0;
      while (emitted < target) {
        const hour = drawHour(rng, persona);
        let t = dayStart(d) + hour * HOUR + Math.floor(rng() * HOUR);
        const ip = !away && persona.altIp && rng.bool(0.15) ? persona.altIp : homeIp;
        const opts = { ip, site, mfa: away ? true : persona.mfa };

        if (rng.bool(0.72)) {
          // A task: several related calls against the same resource.
          const flow = rng.pick(flows);
          const resource = resourceFor(rng, persona);
          for (const action of flow) {
            events.push(baseEvent(rng, persona, t, action, { ...opts, resource }));
            t += rng.intRange(5, 110) * 1000;
            emitted++;
          }
        } else {
          events.push(baseEvent(rng, persona, t, rng.weighted(items, weights), {
            ...opts,
            resource: resourceFor(rng, persona),
          }));
          emitted++;
        }
      }
    }
  }

  benignConfusers(rng, ctx, events);

  // --- Injected campaigns ---------------------------------------------------
  const injected = [];
  for (const id of campaigns) {
    const spec = CAMPAIGNS[id];
    if (!spec) continue;
    const built = spec.build(rng, ctx);
    if (!built || !built.events.length) continue;
    for (const e of built.events) e.campaign = id;
    const times = built.events.map((e) => e.ts);
    injected.push({
      id,
      name: spec.name,
      technique: spec.technique,
      difficulty: spec.difficulty,
      actor: built.actor,
      start: Math.min(...times),
      end: Math.max(...times),
      events: built.events.length,
      malicious: built.events.filter((e) => e.label === 1).length,
    });
    events.push(...built.events);
  }

  events.sort((a, b) => a.ts - b.ts);
  events.forEach((e, i) => { e.id = `s${i}`; });

  return {
    events,
    meta: {
      source: 'synthetic',
      seed,
      days,
      users,
      generatedAt: Date.now(),
      window: [startTs, endMidnight],
      personas: personas.map((p) => ({ id: p.id, team: p.team, site: p.site, type: p.actorType })),
      campaigns: injected,
      labelled: true,
    },
  };
}
