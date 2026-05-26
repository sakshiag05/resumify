We're building a resume builder that real people pay for. The kind where someone googles "resume builder," lands on the site, makes an account, spends 20 minutes filling in their work history, pays nine bucks, downloads a PDF, and sends it to a hiring manager. That whole chain has to work. Every link in it. Not on your computer with mocked data — on Vercel, with a real Postgres database, with actual Stripe charges going through. We're building the whole thing right now, in one shot.

You've shipped stuff like this before so I'm not going to explain what production means. You already know — it means someone's cousin who has never touched a developer tool can use it without breaking it. It means the error states are real, the loading states actually show up, the edge cases don't just throw a 500. So when something has two possible implementations, you pick the one that makes more sense and you write it. You don't leave a comment. You don't write a function that returns undefined and move on. You just build it.

---

**What we're running on**

Next.js 14 App Router. TypeScript strict mode, no wiggle room on that. Tailwind and shadcn/ui handle everything you can see — nothing else gets imported for UI, I don't care how convenient it looks. Zustand holds the editor state. React Query talks to the server. NextAuth handles getting people in — Google, GitHub, and email with a password, all three working for real. Prisma on top of Postgres. API routes inside Next.js. Stripe takes the money. Files live on Vercel Blob or S3, your call. Puppeteer generates PDFs on the server. Resend sends the emails. Zod validates on both ends. It all lives on Vercel when it's done.

---

**Getting in**

Three ways to make an account: Google, GitHub, or an email address and a password they set. Email signup isn't just a form that submits — the verification email actually goes out through Resend, the link inside it actually works, and the account actually gets marked as verified when they click it. Forgot password isn't just a screen — it sends the reset email, the link works, they type a new password, they get back in. First time someone logs in they go through three steps: what's your name, what kind of job are you looking for, how much experience do you have. Then they're in the dashboard. Any page that needs a logged-in user just bounces them to `/login` if the session isn't there. No custom middleware magic, just a redirect.

---

**Dashboard**

Their resumes show up as cards. Each card has a thumbnail of what the resume actually looks like, when it was last saved, and a little badge showing its status. They can kick off a new resume from a blank slate or from one of the templates. Existing resumes can be duplicated, renamed, or deleted from a menu. Delete asks them to confirm before anything gets removed — losing work by accident is one of those things that makes people leave and never come back. Somewhere on the page there's a usage indicator, something like "2 of 3 resumes used." Free users see a nudge toward upgrading that's hard to miss but doesn't take over the whole screen.

---

**The editor**

Everything else in this product is just the stuff around the editor. The editor is the product. Left panel is where they put their information in. Right panel is a live PDF preview that re-renders while they type, held back by a 300ms debounce so it's not thrashing. Every section can be folded up and dragged to a new position using dnd-kit. Here's what the sections are:

Personal info — name, email, phone, where they're located, LinkedIn, GitHub, portfolio, and a photo upload.

Summary — Tiptap rich text editor. 400 characters max. There's a live counter that ticks down as they type so they know how close they are.

Work experience — as many jobs as they've had. Each one: title, company, city, start date, end date, a toggle for "I still work here," and bullets they can add or delete one at a time.

Education — degree, institution, field of study, graduation year, GPA if they want it, honors if they have them.

Skills — tag input broken into categories they define. Languages. Frameworks. Tools. Whatever fits their background.

Projects — title, description, tech stack tags, link to the live thing, link to the code.

Certifications — what it's called, who gave it, when, where to verify it.

Custom section — they name it, they fill it with whatever blocks of content they need.

Toggling a section off doesn't wipe the data. It just hides it until they turn it back on. Undo and redo through Zustand with a minimum of 30 steps. Auto-save fires every 30 seconds and also whenever they click out of any field. There's a last saved timestamp somewhere visible so they don't have to wonder.

---

**Templates**

Five: Classic, Modern, Minimal, Creative, Executive. There's a panel with thumbnails and they can switch between them whenever they want. Switching never loses content — everything comes along every time. Each template lets them adjust: the accent color with six swatches or a hex input, the font across four choices (serif, sans-serif, mono, humanist), the font size scale between compact, normal, and spacious, and the margins between narrow, standard, and wide. All five templates have to actually be ATS-friendly. Single column. Real headings. Nothing that confuses a parser.

---

**AI**

There's an improve button on the summary and on every work bullet individually. Three modes: rewrite it to hit harder, tailor it to a job description they paste in, or fix the grammar. Separate from that there's a JD analyzer — they drop in a job posting and get a score for how well their resume lines up plus a breakdown of what's missing. Every AI suggestion lands in a modal that shows them exactly what changed, and they can take it, skip it, or mess with it before anything on their actual resume moves. Five calls a month free. Unlimited on pro.

---

**Getting the resume out**

PDF is Puppeteer, server-side. Not a screenshot of the page — an actual generated PDF file that looks like the preview. DOCX uses the `docx` package. Every resume has a shareable URL at `/r/[slug]`. That page is read-only, loads clean, has meta tags that work, and has an Open Graph image so it looks like something when shared. Free accounts get a watermark on the PDF. Pro accounts get a clean PDF, get DOCX, and get to set a custom slug.

---

**Money**

Stripe Checkout for upgrading. Stripe Customer Portal for managing or cancelling. Webhook lives at `/api/webhooks/stripe` and covers `checkout.session.completed`, `customer.subscription.updated`, and `customer.subscription.deleted` — all three, handled properly, nothing fails quietly. The billing page shows the plan they're on, when it renews, and their invoice history.

Free: 1 resume, 3 templates, watermarked PDF, 5 AI uses a month.

Pro — $9 a month or $79 a year: unlimited resumes, every template, clean exports, no AI cap, custom slug, priority support.

---

**Settings**

Four tabs. Profile is name, email, photo. Account is change password or delete the whole account — and the delete doesn't go through until they've typed the word DELETE into a box. Notifications is the email toggles for weekly tips and product news. API is their key, but that tab only appears for pro users.

---

**The data**

`sections` on Resume is JSON and it has to look like this:

```ts
{
  personalInfo: {
    name: string, email: string, phone: string, location: string,
    linkedin: string, github: string, portfolio: string, photoUrl: string
  },
  summary: string,
  workExperience: Array<{
    id: string, company: string, position: string, location: string,
    startDate: string, endDate: string, current: boolean, bullets: string[]
  }>,
  education: Array<{
    id: string, institution: string, degree: string, field: string,
    graduationYear: string, gpa?: string, honors?: string
  }>,
  skills: Array<{ id: string, category: string, skills: string[] }>,
  projects: Array<{
    id: string, title: string, description: string,
    techStack: string[], liveUrl: string, githubUrl: string
  }>,
  certifications: Array<{
    id: string, name: string, issuer: string, date: string, url: string
  }>,
  custom?: {
    title: string,
    blocks: Array<{ id: string, title: string, content: string }>
  },
  visibility: Record<string, boolean>,
  order: string[]
}
```

`settings` is also JSON:

```ts
{
  accentColor: string,
  fontFamily: "serif" | "sans-serif" | "mono" | "humanist",
  fontSize: "compact" | "normal" | "spacious",
  margin: "narrow" | "standard" | "wide"
}
```

The seed script makes all five templates and a demo user. Email `demo@resumebuilder.io`, password `Demo1234!`. That user has one Classic resume already built — a software engineer, two real jobs with real-sounding names and actual bullet points that describe real work, one school, three skill categories. Not lorem ipsum. Something that looks like a person who needs a job wrote it.

---

**Database**

`prisma/schema.prisma`:

```
User         — id, email, name, image, passwordHash?, plan, stripeCustomerId, createdAt, updatedAt
Resume       — id, userId, title, slug, templateId, settings(JSON), sections(JSON), isPublic, createdAt, updatedAt
Template     — id, name, thumbnail, isPremium
Subscription — id, userId, stripeSubscriptionId, status, currentPeriodEnd
AiUsage      — id, userId, month, count
Export       — id, resumeId, format, url, createdAt
```

**API routes** — Zod on every one, right status codes:

```
POST   /api/auth/[...nextauth]
GET    /api/resumes
POST   /api/resumes
GET    /api/resumes/[id]
PATCH  /api/resumes/[id]
DELETE /api/resumes/[id]
POST   /api/resumes/[id]/duplicate
POST   /api/resumes/[id]/export/pdf
POST   /api/resumes/[id]/export/docx
GET    /api/templates
POST   /api/ai/improve
POST   /api/ai/analyze-jd
POST   /api/billing/checkout
POST   /api/billing/portal
POST   /api/webhooks/stripe
GET    /api/user/me
PATCH  /api/user/me
DELETE /api/user/me
```

**Pages:**

```
/               — landing page: hero, features, pricing, FAQ, footer
/login
/signup
/verify-email
/dashboard      — protected
/editor/[id]    — protected
/templates      — protected
/billing        — protected
/settings       — protected
/r/[slug]       — public
```

**Folders:**

```
/app
  /(auth)/login, /signup, /verify-email
  /(dashboard)/dashboard, /editor/[id], /templates, /billing, /settings
  /(public)/r/[slug]
  /api/...
/components
  /ui
  /editor     — SectionPanel, LivePreview, TemplateSelector, AIAssistant
  /resume     — Classic.tsx, Modern.tsx, Minimal.tsx, Creative.tsx, Executive.tsx
  /billing    — PricingTable, PlanBadge, UsageMeter
  /dashboard  — ResumeCard, CreateResumeModal
  /shared     — Navbar, Footer, ConfirmModal, EmptyState
/lib
  prisma.ts, auth.ts, stripe.ts, ai.ts, pdf.ts, /validations
/stores
  useResumeStore.ts, useUIStore.ts
/hooks
  useAutoSave.ts, useUndoRedo.ts, useAIUsage.ts
/types
  resume.ts, user.ts, billing.ts
/prisma
  schema.prisma, seed.ts
```

**Env vars in `.env.example`, each one with a note on what it does:**

```
DATABASE_URL
NEXTAUTH_SECRET
NEXTAUTH_URL
GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET
STRIPE_SECRET_KEY / STRIPE_WEBHOOK_SECRET
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
NEXT_PUBLIC_STRIPE_PRO_MONTHLY_PRICE_ID
NEXT_PUBLIC_STRIPE_PRO_YEARLY_PRICE_ID
OPENAI_API_KEY
RESEND_API_KEY
BLOB_READ_WRITE_TOKEN
NEXT_PUBLIC_APP_URL
```

---

**Security**

Session check at the top of every route before anything runs. No session, 401, done. Users only get to their own resumes — anything else is a 403. DOMPurify on the client, same stripping on the server. AI routes are rate limited — Upstash in production, in-memory locally. Stripe webhook signatures get verified on every request. Uploads get their MIME type validated and cap at 5MB. Nothing sensitive touches a `NEXT_PUBLIC_` variable.

---

**Delivering it**

Every file written completely. Path labeled before the code block. Order goes: schema and seed, lib, stores and hooks, API routes, pages, components, README last. If something imports from a file, that file is in the deliverables. Nothing gets cut off. No "I can continue if asked." Finished means finished.

---

**Done when**

No `any` anywhere, strict mode throughout. Typed props on every component. Skeleton on every async operation. Error boundaries around the editor and dashboard. Optimistic saves so nothing feels slow. react-hook-form plus Zod resolver on every form. Layout holds at 375, 768, 1280. Lighthouse 85 performance, 95 accessibility. Stripe webhook handles all three events with no silent failures. PDF export is a real file. Demo user logs in, edits their resume, exports it, nothing breaks. Public page at `/r/[slug]` loads without auth. Nothing missing, nothing stubbed, nothing deferred.

