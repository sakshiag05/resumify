"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { useForm, useFieldArray } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { debounce } from "lodash";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import {
  Button,
} from "@/components/ui/button";
import {
  Input,
} from "@/components/ui/input";
import {
  Textarea,
} from "@/components/ui/textarea";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2, Save } from "lucide-react";

const ResumeSchema = z.object({
  personalInfo: z.object({
    name: z.string().min(2),
    email: z.string().email(),
    phone: z.string(),
    location: z.string(),
    linkedin: z.string(),
    github: z.string(),
    portfolio: z.string(),
    photoUrl: z.string(),
  }),
  summary: z.string().max(400),
  workExperience: z.array(
    z.object({
      id: z.string(),
      company: z.string(),
      position: z.string(),
      location: z.string(),
      startDate: z.string(),
      endDate: z.string(),
      current: z.boolean(),
      bullets: z.array(z.string()),
    })
  ),
});

type ResumeForm = z.infer<typeof ResumeSchema>;

interface ResumeStore {
  history: ResumeForm[];
  currentIndex: number;
  setState: (state: ResumeForm) => void;
  undo: () => ResumeForm | null;
  redo: () => ResumeForm | null;
}

const useResumeStore = create<ResumeStore>()(
  persist(
    (set, get) => ({
      history: [],
      currentIndex: -1,

      setState: (state) => {
        const { history, currentIndex } = get();

        const updated = [...history.slice(0, currentIndex + 1), state].slice(
          -30
        );

        set({
          history: updated,
          currentIndex: updated.length - 1,
        });
      },

      undo: () => {
        const { history, currentIndex } = get();

        if (currentIndex <= 0) return null;

        const newIndex = currentIndex - 1;

        set({
          currentIndex: newIndex,
        });

        return history[newIndex];
      },

      redo: () => {
        const { history, currentIndex } = get();

        if (currentIndex >= history.length - 1) return null;

        const newIndex = currentIndex + 1;

        set({
          currentIndex: newIndex,
        });

        return history[newIndex];
      },
    }),
    {
      name: "resume-store",
    }
  )
);

export default function ResumeEditorPage() {
  const params = useParams();
  const resumeId = params.id as string;

  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<string>("Never");

  const sensors = useSensors(useSensor(PointerSensor));

  const {
    register,
    control,
    watch,
    setValue,
    handleSubmit,
    formState: { errors },
  } = useForm<ResumeForm>({
    resolver: zodResolver(ResumeSchema),
    defaultValues: {
      personalInfo: {
        name: "",
        email: "",
        phone: "",
        location: "",
        linkedin: "",
        github: "",
        portfolio: "",
        photoUrl: "",
      },
      summary: "",
      workExperience: [],
    },
  });

  const {
    fields: workFields,
    append,
    remove,
    move,
  } = useFieldArray({
    control,
    name: "workExperience",
  });

  const watched = watch();

  const saveResume = async (data: ResumeForm) => {
    try {
      setSaving(true);

      const response = await fetch(`/api/resumes/${resumeId}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error("Failed to save");
      }

      setLastSaved(new Date().toLocaleTimeString());
    } catch (error) {
      console.error(error);
    } finally {
      setSaving(false);
    }
  };

  const debouncedSave = useMemo(
    () =>
      debounce(async (data: ResumeForm) => {
        await saveResume(data);
      }, 300),
    []
  );

  useEffect(() => {
    debouncedSave(watched);
  }, [watched]);

  const onSubmit = async (data: ResumeForm) => {
    await saveResume(data);
  };

  return (
    <div className="min-h-screen bg-muted/30">
      <div className="grid lg:grid-cols-2 gap-6 p-6">
        <Card className="h-[95vh] overflow-y-auto">
          <CardContent className="p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold">
                  Resume Editor
                </h1>

                <p className="text-sm text-muted-foreground">
                  Last saved: {lastSaved}
                </p>
              </div>

              <Button
                onClick={handleSubmit(onSubmit)}
                disabled={saving}
              >
                {saving ? (
                  <Loader2 className="animate-spin h-4 w-4" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
              </Button>
            </div>

            <Tabs defaultValue="personal">
              <TabsList className="grid grid-cols-3">
                <TabsTrigger value="personal">
                  Personal
                </TabsTrigger>

                <TabsTrigger value="summary">
                  Summary
                </TabsTrigger>

                <TabsTrigger value="experience">
                  Experience
                </TabsTrigger>
              </TabsList>

              <TabsContent value="personal">
                <div className="space-y-4">
                  <Input
                    placeholder="Full Name"
                    {...register("personalInfo.name")}
                  />

                  <Input
                    placeholder="Email"
                    {...register("personalInfo.email")}
                  />

                  <Input
                    placeholder="Phone"
                    {...register("personalInfo.phone")}
                  />

                  <Input
                    placeholder="Location"
                    {...register("personalInfo.location")}
                  />

                  <Input
                    placeholder="LinkedIn"
                    {...register("personalInfo.linkedin")}
                  />

                  <Input
                    placeholder="GitHub"
                    {...register("personalInfo.github")}
                  />

                  <Input
                    placeholder="Portfolio"
                    {...register("personalInfo.portfolio")}
                  />

                  <Input
                    placeholder="Photo URL"
                    {...register("personalInfo.photoUrl")}
                  />
                </div>
              </TabsContent>

              <TabsContent value="summary">
                <div className="space-y-3">
                  <Textarea
                    rows={8}
                    maxLength={400}
                    placeholder="Write a professional summary..."
                    {...register("summary")}
                  />

                  <div className="flex justify-between text-sm">
                    <span>
                      {watched.summary.length}/400
                    </span>

                    {errors.summary && (
                      <Badge variant="destructive">
                        {errors.summary.message}
                      </Badge>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="experience">
                <div className="space-y-4">
                  <Button
                    type="button"
                    onClick={() =>
                      append({
                        id: crypto.randomUUID(),
                        company: "",
                        position: "",
                        location: "",
                        startDate: "",
                        endDate: "",
                        current: false,
                        bullets: [""],
                      })
                    }
                  >
                    Add Experience
                  </Button>

                  <DndContext
                    sensors={sensors}
                    collisionDetection={closestCenter}
                    onDragEnd={(event) => {
                      const { active, over } = event;

                      if (!over || active.id === over.id) {
                        return;
                      }

                      const oldIndex = workFields.findIndex(
                        (item) => item.id === active.id
                      );

                      const newIndex = workFields.findIndex(
                        (item) => item.id === over.id
                      );

                      move(oldIndex, newIndex);
                    }}
                  >
                    <SortableContext
                      items={workFields}
                      strategy={verticalListSortingStrategy}
                    >
                      {workFields.map((field, index) => (
                        <Card
                          key={field.id}
                          className="border"
                        >
                          <CardContent className="p-4 space-y-3">
                            <Input
                              placeholder="Position"
                              {...register(
                                `workExperience.${index}.position`
                              )}
                            />

                            <Input
                              placeholder="Company"
                              {...register(
                                `workExperience.${index}.company`
                              )}
                            />

                            <Input
                              placeholder="Location"
                              {...register(
                                `workExperience.${index}.location`
                              )}
                            />

                            <div className="grid grid-cols-2 gap-2">
                              <Input
                                type="date"
                                {...register(
                                  `workExperience.${index}.startDate`
                                )}
                              />

                              <Input
                                type="date"
                                {...register(
                                  `workExperience.${index}.endDate`
                                )}
                              />
                            </div>

                            <Button
                              variant="destructive"
                              onClick={() => remove(index)}
                            >
                              Delete
                            </Button>
                          </CardContent>
                        </Card>
                      ))}
                    </SortableContext>
                  </DndContext>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        <Card className="h-[95vh] overflow-hidden">
          <CardContent className="p-0 h-full">
            <iframe
              title="Resume Preview"
              className="w-full h-full bg-white"
              srcDoc={`
                <html>
                  <head>
                    <style>
                      body {
                        font-family: Arial;
                        padding: 40px;
                        color: #111;
                      }

                      h1 {
                        font-size: 32px;
                        margin-bottom: 4px;
                      }

                      h2 {
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 4px;
                        margin-top: 24px;
                      }

                      ul {
                        padding-left: 20px;
                      }
                    </style>
                  </head>

                  <body>
                    <h1>${watched.personalInfo.name}</h1>

                    <p>
                      ${watched.personalInfo.email}
                    </p>

                    <p>
                      ${watched.personalInfo.location}
                    </p>

                    <h2>Summary</h2>

                    <p>${watched.summary}</p>

                    <h2>Experience</h2>

                    ${watched.workExperience
                      .map(
                        (job) => `
                        <div>
                          <strong>${job.position}</strong>
                          — ${job.company}

                          <ul>
                            ${job.bullets
                              .map(
                                (bullet) =>
                                  `<li>${bullet}</li>`
                              )
                              .join("")}
                          </ul>
                        </div>
                      `
                      )
                      .join("")}
                  </body>
                </html>
              `}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
