import Image from "next/image";
import { Loader2} from "Lucide-react";
import { SignIn, ClerkLoaded , ClerkLoading } from '@clerk/nextjs'

export default function Page() {
  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-2">
      <div className="h-full lg:flex flex-col items-center justify-center px-4">
        <div className="text-center space-y-4 pt-16">
          <h1 className="font-bold text-3xl text-[#2E2A47]">Welcome Back !</h1>
          <p className="text-base text-[#7E8CA0]">
            Log in or Create an account to access your Dashboard
          </p>
        </div>
        <div className="flex items-center justify-center mt-8">
         <ClerkLoaded>
          <SignIn/> 
         </ClerkLoaded>
         <ClerkLoading>
          <Loader2 className="animate-spin text-muted-foreground" />
         </ClerkLoading>
         
        </div>
      
      </div>
      <div className="h-full bg-blue-600 hidden lg:flex items-center justify-center">
       <Image src="/LogoSmartCred Blue Background.png" height={500} width={500} alt="Logo" className="w-1/2 h-auto" />
      </div>
    
  </div>
);
}