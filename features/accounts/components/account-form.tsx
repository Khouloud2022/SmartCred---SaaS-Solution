import { z } from "zod";
import { Trash } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { insertAccountSchema } from "@/db/schema";
import {
Form,
FormControl,
FormField,
FormItem,
FormLabel,
FormMessage,
}from "@/components/ui/form";

const formSchema = insertAccountSchema.pick({
    name: true,
});

type FormValues = z.input<typeof formSchema>;

type Props = {
    id?: string;
    defaultValues?: FormValues;
    onSubmit: (values: FormValues) => void;
    onDelete?: () => void;
    disabled?: boolean;
};

export const AccountForm = ({
    id,
    defaultValues,
    onSubmit,
    onDelete,
    disabled,
}: Props) => {
    const form = useForm<FormValues>(
        {
          resolver: zodResolver(formSchema), 
          defaultValues: defaultValues, 
        }
    );

    const handleSubmit = (values: FormValues) => {
        onSubmit(values);
    };

    const handleDelete =() => {
        onDelete?.();
    };

    return(
        <Form {...form}>
            <form 
            onSubmit={form.handleSubmit(handleSubmit)}
            className="space-y-4 pt-4"
            >
                <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Name</FormLabel>
                            <FormControl>
                                <Input
                                    placeholder="e.g, Bank, Credit ..."
                                    disabled={disabled}
                                    {...field}
                                />
                            </FormControl>
                            <FormMessage />
                        </FormItem>
                    )}
                />
                

                <Button 
                 className="w-full"
                 disabled={disabled}>
                    {id ? "Save Changes" : "Create Account"}
                </Button>
                {!!id && (
                    <Button
                        type="button"
                        variant="outline"
                        className="w-full"
                        disabled={disabled}
                        onClick={handleDelete}
                    >
                        <Trash className="size-4 mr-2" />
                        Delete Account
                    </Button>
                )}

            </form>

        </Form>
    )
};