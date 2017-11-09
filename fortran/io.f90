module io
    implicit none

    contains
        integer function read_natm(file_path)
            implicit none

            character(len=1024),intent(in) :: file_path

            integer :: line,natm,iostat
            character(len=8) :: string

            line = 1
            natm = 0

            open(unit=1,status='old',file=file_path,action='read')
            do while(.true.)
                if (line.le.6) then
                    read(unit=1,fmt=*,iostat=iostat) string
                else
                    read(unit=1,fmt=*,iostat=iostat) string
                    if (iostat.lt.0) then
                        !* EOF
                        exit
                    end if
                    natm = natm + 1
                end if
            end do
            close(unit=1)
            read_natm = natm
        end function read_natm

        subroutine readfile(file_path,thread,cell,natm,frac,Z)
            implicit none

            integer,intent(in) :: natm,thread
            character(len=1024),intent(in) :: file_path
            real(8),intent(out) :: cell(:,:)
            real(8),intent(out) :: frac(1:3,1:natm),Z(1:natm)

            integer :: line,iostat,ii
            real(8) :: lx,ly,lz,f1,f2,f3,ZZ
            character(len=8) :: string

            line = 1
            open(unit=1+thread,status='old',file=file_path,action='read',iostat=iostat)
            if (iostat.ne.0) then
                write(*,*) 'fortran/io.f90 error - file :',trim(file_path),'failed to open with :',iostat
                call exit(0)
            end if
            
            do while (.true.)
                if (line.eq.1) then
                    !* header
                    read(unit=1,fmt=*,iostat=iostat) string
                else if ( (line.ge.2).and.(line.le.4) ) then
                    !* cell vectors
                    read(unit=1,fmt=*,iostat=iostat) lx,ly,lz
                    cell(1,line-1) = lx
                    cell(2,line-1) = ly
                    cell(3,line-1) = lz
                else if ( (line.ge.5).and.(line.le.6) ) then
                    !* header read does not parse \n, need some non-escape chars
                    read(unit=1,fmt=*,iostat=iostat) string
                else 
                    !* fractional coordinates and atomic number
                    read(unit=1,fmt=*,iostat=iostat) f1,f2,f3,ZZ
                    if (iostat.lt.0) then
                        !* EOF
                        exit
                    else if ( (line-6).gt.natm ) then
                        write(*,*) 'fortran/io.f90 error - file :',trim(file_path),' has more than ',natm,'atoms '
                        call exit(0)
                    end if
                    frac(1,line-6) = f1 
                    frac(2,line-6) = f2 
                    frac(3,line-6) = f3
                    Z(line-6) = ZZ
                end if

                line = line + 1
            end do
            
            close(unit=1+thread,iostat=iostat)
            if (iostat.ne.0) then
                write(*,*) 'fortran/io.f90 error - file :',trim(file_path),' failed to close with :',iostat
                call exit(0)
            end if

            !* check for fractional coordinates in [0,1]
            do line=1,natm
                do ii=1,3
                    if ( (frac(ii,line).lt.0.0d0).or.(frac(ii,line).gt.1.0d0) ) then
                        write(*,*) 'fortran/io.f90 error - fractional coordinates must be in [0,1] :',trim(file_path)
                        call exit(0)
                    end if
                end do
            end do
        end subroutine readfile
    
end module io
