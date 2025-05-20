"use client";

import { useUser } from '@clerk/nextjs';
import { Button } from "@/components/ui/button";
import { useNewAccount } from "@/features/accounts/hooks/use-new-account";

export default function Home() {
  const { isLoaded } = useUser();
  const { onOpen } = useNewAccount();

  if (!isLoaded) return <div>Loading authentication...</div>;
  return (
   <div>
      <Button onClick={onOpen}>
        Add an account
      </Button>
   </div>
  );
} 