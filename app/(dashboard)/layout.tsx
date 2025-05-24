'use client'; 

import { Suspense } from 'react';
import {Header} from "@/components/header";
import { useUser } from '@clerk/nextjs';

type Props = {
    children: React.ReactNode;
};

const DashboardLayout = ({children}: Props) => {
     const { isLoaded } = useUser();

  if (!isLoaded) {
    return <div className="flex items-center justify-center h-screen">Loading...</div>;
  }
    
    return (
        <>
            <Header />
            <main className="bg-gray-100 min-h-screen flex flex-col items-center justify-center p-4 lg:p-14 pb-36">
               <Suspense fallback={<div>Loading...</div>}>
                {children}
               </Suspense>
               
            </main>
        </>
    );
};

export default DashboardLayout;
