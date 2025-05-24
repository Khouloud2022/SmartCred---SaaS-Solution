import { pgTable, text, date, integer, decimal } from "drizzle-orm/pg-core";
import { createInsertSchema }  from "drizzle-zod";

export const accounts = pgTable ("accounts", {
    id: text("id").primaryKey(),
    plaidId: text("plaid_id"),
    name: text("name").notNull(),
    userId: text("user_id").notNull(),
});

export const applicants = pgTable("applicants", {
  id: text("id").primaryKey(),
  empTitle: text("emp_title"),
  empLength: integer("emp_length"),
  state: text("state"),
  homeownership: text("homeownership"),
  annualIncome: decimal("annual_income", {
    precision: 10,
    scale: 2,
  }),
  verifiedIncome: text("verified_income"),
  debtToIncome: decimal("debt_to_income", {
    precision: 5,
    scale: 2,
  }),
});

export const loans = pgTable("loans", {
  id: text("id").primaryKey(),
  applicantId: text("applicant_id").references(() => applicants.id),
  loanPurpose: text("loan_purpose"),
  applicationType: text("application_type"),
  loanAmount: decimal("loan_amount", {
    precision: 10,
    scale: 2,
  }),
  term: integer("term"),
  interestRate: decimal("interest_rate", {
    precision: 5,
    scale: 2,
  }),
  installment: decimal("installment", {
    precision: 10,
    scale: 2,
  }),
  grade: text("grade"),
  subGrade: text("sub_grade"),
  issueMonth: date("issue_month"),
  loanStatus: text("loan_status"),
});

export const payments = pgTable("payments", {
  id: text("id").primaryKey(),
  loanId: text("loan_id").references(() => loans.id),
  paymentDate: date("payment_date"),
  paymentAmount: decimal("payment_amount", {
    precision: 10,
    scale: 2,
  }),
  paymentStatus: text("payment_status"),
});


export const insertPaymentSchema = createInsertSchema(payments);
export const insertLoanSchema = createInsertSchema(loans);
export const insertApplicantSchema = createInsertSchema(applicants);
export const insertAccountSchema = createInsertSchema(accounts);