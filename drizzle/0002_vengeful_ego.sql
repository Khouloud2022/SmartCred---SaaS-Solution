CREATE TABLE "applicants" (
	"id" text PRIMARY KEY NOT NULL,
	"emp_title" text,
	"emp_length" integer,
	"state" text,
	"homeownership" text,
	"annual_income" numeric(10, 2),
	"verified_income" text,
	"debt_to_income" numeric(5, 2)
);
--> statement-breakpoint
CREATE TABLE "loans" (
	"id" text PRIMARY KEY NOT NULL,
	"applicant_id" text,
	"loan_purpose" text,
	"application_type" text,
	"loan_amount" numeric(10, 2),
	"term" integer,
	"interest_rate" numeric(5, 2),
	"installment" numeric(10, 2),
	"grade" text,
	"sub_grade" text,
	"issue_month" date,
	"loan_status" text
);
--> statement-breakpoint
CREATE TABLE "payments" (
	"id" text PRIMARY KEY NOT NULL,
	"loan_id" text,
	"payment_date" date,
	"payment_amount" numeric(10, 2),
	"payment_status" text
);
--> statement-breakpoint
ALTER TABLE "loans" ADD CONSTRAINT "loans_applicant_id_applicants_id_fk" FOREIGN KEY ("applicant_id") REFERENCES "public"."applicants"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "payments" ADD CONSTRAINT "payments_loan_id_loans_id_fk" FOREIGN KEY ("loan_id") REFERENCES "public"."loans"("id") ON DELETE no action ON UPDATE no action;