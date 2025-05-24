"use client";

import { useNewAccount } from "@/features/accounts/hooks/use-new-account";
import { useGetAccounts } from "@/features/accounts/api/use-get-accounts";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
Card,
CardContent,
CardHeader,
CardTitle,
} from "@/components/ui/card";
import { DataTable } from "@/components/ui/data-table";

import { columns } from "./columns";




const AccountsPage = () => {

    const newAccount = useNewAccount();
    const accountsQuery = useGetAccounts();
    const accounts = accountsQuery.data?.data || [];

    return (
        <div className="max-w-screen-2xl mx-auto w-full pb-10 -mt-40">
            <Card className="border-none drop-shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between gap-y-2">
                <CardTitle className="text-xl line-clamp-1">
                Accounts
                </CardTitle>
                <Button 
                onClick={newAccount.onOpen} 
                size="sm"
                className="ml-auto" 
                >
                <Plus className="size-4 mr-2" />
                Add Account
                </Button>
            </CardHeader>
            <CardContent>
                <DataTable 
                    filterKey="email"
                    columns={columns} 
                    data={accounts}
      
                    onDelete={()=>{}}
                    disabled={false}
                    />
            </CardContent>
            </Card>
        
        </div>
    );
};

export default AccountsPage;